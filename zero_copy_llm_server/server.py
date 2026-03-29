"""
server.py — persistent LLM inference server.

The model is loaded ONCE and kept resident in memory.
Requests arrive over a UNIX domain socket; each connection carries one prompt
and receives one response.  A special SENTINEL message triggers graceful shutdown.

Usage:
    # start in foreground (Ctrl-C to stop)
    python server.py

    # start as background daemon
    python server.py --daemon

    # stop a running daemon
    python server.py --stop

    # show server status
    python server.py --status

    # custom model path / socket
    python server.py --model /data/models/mymodel.gguf --sock /tmp/myserver.sock
"""
import argparse
import logging
import os
import signal
import socket
import struct
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [server] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PID_FILE = "/tmp/llm_server.pid"


# ── core server loop ──────────────────────────────────────────────

def _load_model(model_path: str, n_ctx: int, n_threads: int,
                n_batch: int, n_gpu_layers: int):
    try:
        from llama_cpp import Llama
    except ImportError:
        log.error("llama-cpp-python not installed.  Run: pip install llama-cpp-python")
        sys.exit(1)

    log.info(f"Loading model: {model_path}")
    log.info(f"  threads={n_threads}  ctx={n_ctx}  batch={n_batch}  gpu_layers={n_gpu_layers}")
    t0 = time.perf_counter()
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    log.info(f"Model loaded in {elapsed:.1f}s")
    return llm


def _recv_all(conn: socket.socket, length: int) -> bytes:
    buf = b""
    while len(buf) < length:
        chunk = conn.recv(length - len(buf))
        if not chunk:
            raise ConnectionError("connection closed mid-read")
        buf += chunk
    return buf


def serve(model_path: str, sock_path: str, n_ctx: int, n_threads: int,
          n_batch: int, n_gpu_layers: int, max_tokens: int, stop_words: list):
    """Main server loop.  Blocks until SENTINEL is received."""
    from config import SENTINEL

    llm = _load_model(model_path, n_ctx, n_threads, n_batch, n_gpu_layers)

    # smoke test
    t0 = time.perf_counter()
    r  = llm("Q: 1+1=? A:", max_tokens=4, stop=["\n"], echo=False)
    log.info(f"Smoke test OK  ({(time.perf_counter()-t0)*1000:.0f} ms): "
             f"{r['choices'][0]['text'].strip()!r}")

    if os.path.exists(sock_path):
        os.unlink(sock_path)

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(64)
    log.info(f"Listening on {sock_path}")

    request_count = 0
    try:
        while True:
            conn, _ = srv.accept()
            try:
                raw = conn.recv(4)
                if not raw:
                    conn.close()
                    continue
                length = struct.unpack(">I", raw)[0]
                data   = _recv_all(conn, length)

                if data == SENTINEL:
                    log.info("Received SENTINEL — shutting down")
                    conn.close()
                    break

                prompt = data.decode("utf-8")
                t0     = time.perf_counter()
                resp   = llm(prompt, max_tokens=max_tokens,
                             stop=stop_words, echo=False)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                text   = resp["choices"][0]["text"].strip()
                tokens = resp["usage"]["completion_tokens"]

                request_count += 1
                log.info(f"[req {request_count:04d}] {elapsed_ms:.0f}ms  "
                         f"{tokens}tok  {tokens/(elapsed_ms/1000):.1f}tok/s  "
                         f"prompt={prompt[:40]!r}")

                encoded = text.encode("utf-8")
                conn.sendall(struct.pack(">I", len(encoded)) + encoded)
            except Exception as e:
                log.warning(f"Request error: {e}")
            finally:
                conn.close()
    finally:
        srv.close()
        if os.path.exists(sock_path):
            os.unlink(sock_path)
        log.info(f"Server stopped. Total requests served: {request_count}")


# ── daemon helpers ────────────────────────────────────────────────

def _start_daemon(args):
    """Fork into background and write PID file."""
    pid = os.fork()
    if pid > 0:
        # parent: write pid and exit
        with open(PID_FILE, "w") as f:
            f.write(str(pid))
        print(f"Server started (pid={pid})")
        print(f"Logs: check /tmp/llm_server.log")
        print(f"Stop: python server.py --stop")
        sys.exit(0)

    # child: redirect stdio, run server
    os.setsid()
    with open("/tmp/llm_server.log", "a") as logf:
        os.dup2(logf.fileno(), sys.stdout.fileno())
        os.dup2(logf.fileno(), sys.stderr.fileno())

    serve(
        model_path   = args.model,
        sock_path    = args.sock,
        n_ctx        = args.n_ctx,
        n_threads    = args.n_threads,
        n_batch      = args.n_batch,
        n_gpu_layers = args.n_gpu_layers,
        max_tokens   = args.max_tokens,
        stop_words   = args.stop.split(","),
    )


def _stop_daemon():
    if not os.path.exists(PID_FILE):
        print("No PID file found — server may not be running")
        return
    with open(PID_FILE) as f:
        pid = int(f.read().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to pid {pid}")
    except ProcessLookupError:
        print(f"Process {pid} not found")
    finally:
        os.unlink(PID_FILE)


def _status():
    if not os.path.exists(PID_FILE):
        print("Status: stopped (no PID file)")
        return
    with open(PID_FILE) as f:
        pid = int(f.read().strip())
    try:
        os.kill(pid, 0)
        print(f"Status: running (pid={pid})")
    except ProcessLookupError:
        print(f"Status: stale PID file (pid={pid} not found)")


# ── CLI ───────────────────────────────────────────────────────────

def main():
    from config import (MODEL_PATH, UNIX_SOCK_PATH, N_CTX, N_THREADS,
                        N_BATCH, N_GPU_LAYERS, MAX_TOKENS)

    parser = argparse.ArgumentParser(description="Persistent LLM inference server")
    parser.add_argument("--model",       default=MODEL_PATH,     help="path to GGUF file")
    parser.add_argument("--sock",        default=UNIX_SOCK_PATH, help="UNIX socket path")
    parser.add_argument("--n-ctx",       type=int, default=N_CTX)
    parser.add_argument("--n-threads",   type=int, default=N_THREADS)
    parser.add_argument("--n-batch",     type=int, default=N_BATCH)
    parser.add_argument("--n-gpu-layers",type=int, default=N_GPU_LAYERS)
    parser.add_argument("--max-tokens",  type=int, default=MAX_TOKENS)
    parser.add_argument("--stop",        default="\n,Q:",        help="comma-separated stop words")
    parser.add_argument("--daemon",      action="store_true",    help="run as background daemon")
    parser.add_argument("--stop-server", action="store_true",    help="stop running daemon")
    parser.add_argument("--status",      action="store_true",    help="show server status")
    args = parser.parse_args()

    if args.status:
        _status(); return

    if args.stop_server:
        _stop_daemon(); return

    if not os.path.exists(args.model):
        print(f"ERROR: model not found: {args.model}")
        print("Run:  python download_model.py")
        sys.exit(1)

    if args.daemon:
        _start_daemon(args)
    else:
        serve(
            model_path   = args.model,
            sock_path    = args.sock,
            n_ctx        = args.n_ctx,
            n_threads    = args.n_threads,
            n_batch      = args.n_batch,
            n_gpu_layers = args.n_gpu_layers,
            max_tokens   = args.max_tokens,
            stop_words   = args.stop.split(","),
        )


if __name__ == "__main__":
    main()
