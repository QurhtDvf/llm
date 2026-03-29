"""
benchmark.py — compare 3 IPC methods for LLM inference.

  Method 1 (Baseline)  : pickle serialization over TCP loopback
  Method 2 (Zero-copy) : shared memory + memoryview, spin-poll control flag
  Method 3 (Persistent): model resident in one process, UNIX domain socket

Usage:
    python benchmark.py                      # run all 3, save chart
    python benchmark.py --methods 2 3        # run only methods 2 and 3
    python benchmark.py --n 5 --tokens 32    # 5 requests, 32 max tokens
    python benchmark.py --no-chart           # skip matplotlib output
"""
import argparse
import gc
import logging
import os
import pickle
import socket
import struct
import sys
import time
from multiprocessing import Process, Pipe, Event, shared_memory

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [bench] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── shared helpers ────────────────────────────────────────────────

def _recv_all(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed")
        buf += chunk
    return buf


def _unlink_shm(name: str):
    try:
        s = shared_memory.SharedMemory(name=name)
        s.close(); s.unlink()
    except Exception:
        pass


def _load_llm(model_path, n_ctx, n_threads, n_batch, n_gpu_layers):
    from llama_cpp import Llama
    return Llama(
        model_path=model_path, n_ctx=n_ctx,
        n_threads=n_threads, n_batch=n_batch,
        n_gpu_layers=n_gpu_layers, verbose=False,
    )


# ── Method 1: pickle + TCP ────────────────────────────────────────

def _tcp_server(model_path, n_ctx, n_threads, n_batch, n_gpu_layers,
                max_tokens, stop_words, port, ready_event, n):
    llm = _load_llm(model_path, n_ctx, n_threads, n_batch, n_gpu_layers)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port)); srv.listen(1)
    ready_event.set()
    conn, _ = srv.accept()
    for _ in range(n):
        raw = conn.recv(4)
        if not raw: break
        data   = _recv_all(conn, struct.unpack(">I", raw)[0])
        prompt = pickle.loads(data)
        resp   = llm(prompt, max_tokens=max_tokens, stop=stop_words, echo=False)
        out    = pickle.dumps(resp["choices"][0]["text"].strip())
        conn.sendall(struct.pack(">I", len(out)) + out)
    conn.close(); srv.close()


def benchmark_tcp(cfg) -> tuple[list[float], list[str]]:
    log.info("Method 1: pickle + TCP loopback  (model loads fresh each run)")
    ready = Event()
    p = Process(target=_tcp_server, args=(
        cfg.model_path, cfg.n_ctx, cfg.n_threads, cfg.n_batch, cfg.n_gpu_layers,
        cfg.max_tokens, cfg.stop_words, cfg.tcp_port, ready, cfg.n_requests,
    ))
    p.start(); ready.wait()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", cfg.tcp_port))

    latencies, results = [], []
    for i in range(cfg.n_requests):
        prompt  = cfg.test_prompts[i % len(cfg.test_prompts)]
        t0      = time.perf_counter()
        payload = pickle.dumps(prompt)
        sock.sendall(struct.pack(">I", len(payload)) + payload)
        data = _recv_all(sock, struct.unpack(">I", sock.recv(4))[0])
        latencies.append((time.perf_counter() - t0) * 1000)
        results.append(pickle.loads(data))
        log.info(f"  [{i+1}/{cfg.n_requests}] {latencies[-1]:.0f} ms  {results[-1]!r}")

    sock.close(); p.join()
    return latencies, results


# ── Method 2: shared memory + zero-copy ──────────────────────────

def _shm_server(model_path, n_ctx, n_threads, n_batch, n_gpu_layers,
                max_tokens, stop_words,
                shm_in_name, shm_out_name, shm_ctrl_name,
                max_prompt, max_result, n):
    llm   = _load_llm(model_path, n_ctx, n_threads, n_batch, n_gpu_layers)
    s_in  = shared_memory.SharedMemory(name=shm_in_name)
    s_out = shared_memory.SharedMemory(name=shm_out_name)
    s_ctl = shared_memory.SharedMemory(name=shm_ctrl_name)
    mv_in = memoryview(s_in.buf)
    mv_out= memoryview(s_out.buf)
    cb    = s_ctl.buf
    for _ in range(n):
        while cb[0] != 1: pass                          # spin-wait
        p_len  = struct.unpack_from(">I", mv_in, 0)[0]
        prompt = bytes(mv_in[4:4+p_len]).decode("utf-8")
        resp   = llm(prompt, max_tokens=max_tokens, stop=stop_words, echo=False)
        text   = resp["choices"][0]["text"].strip()
        r_b    = text.encode("utf-8")
        struct.pack_into(">I", mv_out, 0, len(r_b))
        mv_out[4:4+len(r_b)] = r_b
        cb[0] = 2                                       # response ready
    del mv_in, mv_out
    s_in.close(); s_out.close(); s_ctl.close()


def benchmark_shm(cfg) -> tuple[list[float], list[str]]:
    from config import SHM_INPUT_NAME, SHM_OUTPUT_NAME, SHM_CTRL_NAME, SHM_TOTAL

    log.info("Method 2: shared memory + zero-copy  (0 data copies)")
    for name in [SHM_INPUT_NAME, SHM_OUTPUT_NAME, SHM_CTRL_NAME]:
        _unlink_shm(name)

    s_in  = shared_memory.SharedMemory(create=True, size=SHM_TOTAL,           name=SHM_INPUT_NAME)
    s_out = shared_memory.SharedMemory(create=True, size=cfg.max_result+4,    name=SHM_OUTPUT_NAME)
    s_ctl = shared_memory.SharedMemory(create=True, size=1,                   name=SHM_CTRL_NAME)
    s_ctl.buf[0] = 0
    mv_in = memoryview(s_in.buf)
    mv_out= memoryview(s_out.buf)
    cb    = s_ctl.buf

    p = Process(target=_shm_server, args=(
        cfg.model_path, cfg.n_ctx, cfg.n_threads, cfg.n_batch, cfg.n_gpu_layers,
        cfg.max_tokens, cfg.stop_words,
        SHM_INPUT_NAME, SHM_OUTPUT_NAME, SHM_CTRL_NAME,
        cfg.max_prompt, cfg.max_result, cfg.n_requests,
    ))
    p.start()

    latencies, results = [], []
    for i in range(cfg.n_requests):
        prompt = cfg.test_prompts[i % len(cfg.test_prompts)]
        p_b    = prompt.encode("utf-8")
        t0     = time.perf_counter()
        struct.pack_into(">I", mv_in, 0, len(p_b))
        mv_in[4:4+len(p_b)] = p_b
        cb[0] = 1
        while cb[0] != 2: pass
        r_len  = struct.unpack_from(">I", mv_out, 0)[0]
        result = bytes(mv_out[4:4+r_len]).decode("utf-8")
        cb[0]  = 0
        latencies.append((time.perf_counter() - t0) * 1000)
        results.append(result)
        log.info(f"  [{i+1}/{cfg.n_requests}] {latencies[-1]:.0f} ms  {result!r}")

    p.join()
    del mv_in, mv_out
    for s in [s_in, s_out, s_ctl]:
        try: s.close(); s.unlink()
        except Exception: pass
    gc.collect()
    return latencies, results


# ── Method 3: persistent server + UNIX socket ─────────────────────

def _persistent_server(model_path, n_ctx, n_threads, n_batch, n_gpu_layers,
                        max_tokens, stop_words, sock_path, ready_pipe):
    from config import SENTINEL
    llm = _load_llm(model_path, n_ctx, n_threads, n_batch, n_gpu_layers)
    if os.path.exists(sock_path): os.unlink(sock_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path); srv.listen(64)
    ready_pipe.send(True)
    while True:
        conn, _ = srv.accept()
        try:
            raw = conn.recv(4)
            if not raw: conn.close(); continue
            data = _recv_all(conn, struct.unpack(">I", raw)[0])
            if data == SENTINEL:
                conn.close(); break
            prompt = data.decode("utf-8")
            resp   = llm(prompt, max_tokens=max_tokens, stop=stop_words, echo=False)
            text   = resp["choices"][0]["text"].strip().encode("utf-8")
            conn.sendall(struct.pack(">I", len(text)) + text)
        except Exception as e:
            log.warning(f"server error: {e}")
        finally:
            conn.close()
    srv.close()
    if os.path.exists(sock_path): os.unlink(sock_path)


def _unix_call(sock_path, prompt):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock_path)
    d = prompt.encode("utf-8")
    s.sendall(struct.pack(">I", len(d)) + d)
    raw = s.recv(4)
    buf = _recv_all(s, struct.unpack(">I", raw)[0])
    s.close()
    return buf.decode("utf-8")


def benchmark_persistent(cfg) -> tuple[list[float], list[str]]:
    from config import SENTINEL, UNIX_SOCK_PATH

    log.info("Method 3: persistent server + UNIX socket  (model loaded once)")
    parent, child = Pipe()
    p = Process(target=_persistent_server, args=(
        cfg.model_path, cfg.n_ctx, cfg.n_threads, cfg.n_batch, cfg.n_gpu_layers,
        cfg.max_tokens, cfg.stop_words, UNIX_SOCK_PATH, child,
    ))
    p.start(); parent.recv()

    latencies, results = [], []
    for i in range(cfg.n_requests):
        prompt = cfg.test_prompts[i % len(cfg.test_prompts)]
        t0     = time.perf_counter()
        result = _unix_call(UNIX_SOCK_PATH, prompt)
        latencies.append((time.perf_counter() - t0) * 1000)
        results.append(result)
        log.info(f"  [{i+1}/{cfg.n_requests}] {latencies[-1]:.0f} ms  {result!r}")

    # graceful shutdown
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(UNIX_SOCK_PATH)
    s.sendall(struct.pack(">I", len(SENTINEL)) + SENTINEL)
    s.close()
    p.join()
    return latencies, results


# ── reporting ─────────────────────────────────────────────────────

def print_summary(results_map: dict, n: int):
    print("\n" + "="*65)
    print(f"{'Method':<28}  {'median':>8}  {'p99':>8}  {'qps':>8}  {'speedup':>8}")
    print("-"*65)
    base = None
    for name, lats in results_map.items():
        m   = np.median(lats)
        p99 = np.percentile(lats, 99)
        q   = n / (sum(lats) / 1000)
        if base is None:
            base = m
        spd = base / m
        print(f"{name:<28}  {m:>7.0f}ms  {p99:>7.0f}ms  {q:>8.4f}  {spd:>7.2f}x")
    print("="*65)


def save_chart(results_map: dict, n: int, output_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping chart")
        return

    colors = ["#ef4444", "#22c55e", "#3b82f6"]
    names  = list(results_map.keys())
    lats   = list(results_map.values())

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor("#0f172a")
    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        ax.spines[:].set_color("#334155")

    medians = [np.median(l) for l in lats]
    ax = axes[0]
    bars = ax.bar(names, medians, color=colors, alpha=0.85, edgecolor="#475569")
    for bar, val in zip(bars, medians):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                f"{val:.0f} ms", ha="center", va="bottom", fontsize=9, color="white")
    ax.set_ylabel("Median latency (ms)", color="#94a3b8")
    ax.set_title("Median latency per request", color="#f1f5f9", fontweight="bold")
    ax.tick_params(axis="x", labelsize=8, colors="#cbd5e1")

    qps = [n / (sum(l) / 1000) for l in lats]
    ax  = axes[1]
    bars = ax.bar(names, qps, color=colors, alpha=0.85, edgecolor="#475569")
    for bar, val in zip(bars, qps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, color="white")
    ax.set_ylabel("Throughput (req/s)", color="#94a3b8")
    ax.set_title("Throughput (QPS)", color="#f1f5f9", fontweight="bold")
    ax.tick_params(axis="x", labelsize=8, colors="#cbd5e1")

    plt.suptitle("LLM Inference Server Benchmark  (Qwen2.5-7B Q4_K_M / CPU)",
                 color="#f8fafc", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    log.info(f"Chart saved: {output_path}")


# ── main ──────────────────────────────────────────────────────────

class _Cfg:
    pass


def main():
    from config import (MODEL_PATH, N_CTX, N_THREADS, N_BATCH, N_GPU_LAYERS,
                        MAX_TOKENS, N_REQUESTS, TEST_PROMPTS,
                        TCP_PORT, MAX_PROMPT_BYTES, MAX_RESULT_BYTES,
                        BENCHMARK_OUTPUT)

    parser = argparse.ArgumentParser(description="LLM IPC benchmark")
    parser.add_argument("--model",    default=MODEL_PATH)
    parser.add_argument("--n",        type=int, default=N_REQUESTS,  help="requests per method")
    parser.add_argument("--tokens",   type=int, default=MAX_TOKENS,  help="max tokens to generate")
    parser.add_argument("--ctx",      type=int, default=N_CTX)
    parser.add_argument("--threads",  type=int, default=N_THREADS)
    parser.add_argument("--methods",  type=int, nargs="+", default=[1, 2, 3],
                        help="which methods to run (1=tcp, 2=shm, 3=persistent)")
    parser.add_argument("--no-chart", action="store_true")
    parser.add_argument("--output",   default=BENCHMARK_OUTPUT)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: model not found: {args.model}")
        print("Run:  python download_model.py")
        sys.exit(1)

    cfg = _Cfg()
    cfg.model_path   = args.model
    cfg.n_ctx        = args.ctx
    cfg.n_threads    = args.threads
    cfg.n_batch      = N_BATCH
    cfg.n_gpu_layers = N_GPU_LAYERS
    cfg.max_tokens   = args.tokens
    cfg.stop_words   = ["\n", "Q:"]
    cfg.n_requests   = args.n
    cfg.test_prompts = TEST_PROMPTS
    cfg.tcp_port     = TCP_PORT
    cfg.max_prompt   = MAX_PROMPT_BYTES
    cfg.max_result   = MAX_RESULT_BYTES

    results_map = {}

    if 1 in args.methods:
        lats, _ = benchmark_tcp(cfg)
        results_map["Baseline\npickle+TCP"] = lats

    if 2 in args.methods:
        lats, _ = benchmark_shm(cfg)
        results_map["SharedMem\nZero-Copy"] = lats

    if 3 in args.methods:
        lats, _ = benchmark_persistent(cfg)
        results_map["Persistent\nUnix Socket"] = lats

    if not results_map:
        print("No methods ran."); return

    print_summary(results_map, args.n)

    if not args.no_chart:
        save_chart(results_map, args.n, args.output)


if __name__ == "__main__":
    main()
