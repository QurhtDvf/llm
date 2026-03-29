"""
client.py — client library and CLI for the persistent LLM server.

Library usage:
    from client import LLMClient

    with LLMClient() as client:
        result = client.infer("Q: What is the capital of France? A:")
        print(result)

CLI usage:
    python client.py "Q: What is the capital of France? A:"
    python client.py --sock /tmp/llm_persist.sock "Tell me a joke"
    python client.py --stop              # send shutdown signal to server
    python client.py --bench 10          # send 10 requests and print latencies
"""
import argparse
import socket
import struct
import sys
import time


# ── core send/receive ─────────────────────────────────────────────

def _send_recv(sock_path: str, payload: bytes, timeout: float = 120.0) -> bytes:
    """Send a length-prefixed message and receive a length-prefixed reply."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(sock_path)
        s.sendall(struct.pack(">I", len(payload)) + payload)
        raw_len = s.recv(4)
        if not raw_len:
            raise ConnectionError("server closed connection without response")
        length = struct.unpack(">I", raw_len)[0]
        buf = b""
        while len(buf) < length:
            chunk = s.recv(length - len(buf))
            if not chunk:
                raise ConnectionError("connection closed mid-read")
            buf += chunk
        return buf
    finally:
        s.close()


# ── client class ─────────────────────────────────────────────────

class LLMClient:
    """
    Simple client for the persistent LLM server.

    Usage (context manager):
        with LLMClient() as c:
            text = c.infer("Q: Hello? A:")

    Usage (manual):
        c = LLMClient(sock_path="/tmp/llm_persist.sock")
        text = c.infer("Q: Hello? A:")
    """

    def __init__(self, sock_path: str = None, timeout: float = 120.0):
        if sock_path is None:
            from config import UNIX_SOCK_PATH
            sock_path = UNIX_SOCK_PATH
        self.sock_path = sock_path
        self.timeout   = timeout

    def infer(self, prompt: str) -> str:
        """Send prompt, return generated text."""
        result = _send_recv(
            self.sock_path,
            prompt.encode("utf-8"),
            self.timeout,
        )
        return result.decode("utf-8")

    def infer_timed(self, prompt: str) -> tuple[str, float]:
        """Returns (text, latency_ms)."""
        t0     = time.perf_counter()
        result = self.infer(prompt)
        return result, (time.perf_counter() - t0) * 1000

    def stop_server(self):
        """Send graceful shutdown signal."""
        from config import SENTINEL
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(self.sock_path)
        s.sendall(struct.pack(">I", len(SENTINEL)) + SENTINEL)
        s.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


# ── CLI ───────────────────────────────────────────────────────────

def main():
    from config import UNIX_SOCK_PATH, TEST_PROMPTS

    parser = argparse.ArgumentParser(description="LLM inference client")
    parser.add_argument("prompt",   nargs="?", default=None, help="prompt string")
    parser.add_argument("--sock",   default=UNIX_SOCK_PATH,  help="UNIX socket path")
    parser.add_argument("--stop",   action="store_true",     help="send shutdown to server")
    parser.add_argument("--bench",  type=int, default=0,     help="run N requests and report latencies")
    parser.add_argument("--timeout",type=float, default=120.0)
    args = parser.parse_args()

    client = LLMClient(sock_path=args.sock, timeout=args.timeout)

    if args.stop:
        client.stop_server()
        print("Shutdown signal sent.")
        return

    if args.bench > 0:
        print(f"Benchmarking {args.bench} requests ...")
        latencies = []
        for i in range(args.bench):
            prompt = TEST_PROMPTS[i % len(TEST_PROMPTS)]
            text, ms = client.infer_timed(prompt)
            latencies.append(ms)
            print(f"  [{i+1:03d}] {ms:7.0f} ms  {text[:60]!r}")
        import statistics
        print(f"\nMedian : {statistics.median(latencies):.0f} ms")
        print(f"Mean   : {statistics.mean(latencies):.0f} ms")
        print(f"Min    : {min(latencies):.0f} ms")
        print(f"Max    : {max(latencies):.0f} ms")
        return

    if args.prompt is None:
        parser.print_help()
        sys.exit(1)

    text, ms = client.infer_timed(args.prompt)
    print(f"{text}")
    print(f"[{ms:.0f} ms]", file=sys.stderr)


if __name__ == "__main__":
    main()
