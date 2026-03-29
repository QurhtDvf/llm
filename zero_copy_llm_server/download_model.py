"""
download_model.py — download the GGUF model from Hugging Face.

Usage:
    python download_model.py
    python download_model.py --repo bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
                             --file Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
"""
import argparse
import os
import sys


def download(repo: str, filename: str, local_dir: str) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    os.makedirs(local_dir, exist_ok=True)
    dest = os.path.join(local_dir, filename)

    if os.path.exists(dest):
        size_gb = os.path.getsize(dest) / 1e9
        print(f"[cache] {dest}  ({size_gb:.2f} GB)")
        return dest

    print(f"[download] {repo}/{filename}")
    print("  This may take several minutes on first run (~4.5 GB) ...")
    path = hf_hub_download(repo_id=repo, filename=filename, local_dir=local_dir)
    size_gb = os.path.getsize(path) / 1e9
    print(f"[ok] saved to {path}  ({size_gb:.2f} GB)")
    return path


def main():
    from config import MODEL_REPO, MODEL_FILE, MODEL_DIR

    parser = argparse.ArgumentParser(description="Download GGUF model")
    parser.add_argument("--repo",  default=MODEL_REPO, help="HuggingFace repo id")
    parser.add_argument("--file",  default=MODEL_FILE, help="GGUF filename")
    parser.add_argument("--dir",   default=MODEL_DIR,  help="local directory")
    args = parser.parse_args()

    path = download(args.repo, args.file, args.dir)
    print(f"\nModel path: {path}")
    print("Next step : python server.py --start")


if __name__ == "__main__":
    main()
