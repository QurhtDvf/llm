"""
config.py — central configuration for all modules.
Edit this file to change model, paths, and benchmark parameters.
"""
import os

# ── Model ─────────────────────────────────────────────────────────
MODEL_REPO = "bartowski/Qwen2.5-7B-Instruct-GGUF"
MODEL_FILE = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"

# Alternative models:
# MODEL_REPO = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
# MODEL_FILE = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

# ── Inference ─────────────────────────────────────────────────────
N_THREADS   = os.cpu_count() or 4   # set to physical core count
N_CTX       = 512                   # context window (shorter = faster)
N_BATCH     = 512                   # batch size for prompt processing
MAX_TOKENS  = 64                    # max tokens to generate per request
N_GPU_LAYERS = 0                    # 0 = CPU only; set ~35 for GPU

# ── IPC / sockets ─────────────────────────────────────────────────
UNIX_SOCK_PATH  = "/tmp/llm_persist.sock"
TCP_HOST        = "127.0.0.1"
TCP_PORT        = 19300
SENTINEL        = b"__STOP__"       # graceful shutdown signal

# ── Shared memory ─────────────────────────────────────────────────
MAX_PROMPT_BYTES = 2048
MAX_RESULT_BYTES = 4096
SHM_INPUT_NAME   = "llm_shm_in"
SHM_OUTPUT_NAME  = "llm_shm_out"
SHM_CTRL_NAME    = "llm_shm_ctrl"
# layout: [4B len][prompt bytes][4B len][result bytes]
SHM_TOTAL        = 4 + MAX_PROMPT_BYTES + 4 + MAX_RESULT_BYTES

# ── Benchmark ─────────────────────────────────────────────────────
N_REQUESTS = 3
TEST_PROMPTS = [
    "Q: What is the capital of Japan? A:",
    "Q: What is 2+2? A:",
    "Q: What is the chemical formula of water? A:",
]
BENCHMARK_OUTPUT = "llm_benchmark.png"
