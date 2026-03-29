#!/usr/bin/env bash
# setup.sh — EC2 initial setup (Amazon Linux 2023 / Ubuntu 22.04)
# Usage: bash setup.sh
set -euo pipefail

echo "=== Zero-Copy LLM Server — EC2 Setup ==="

# ── detect OS ────────────────────────────────────────────────────
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS_ID="$ID"
else
    OS_ID="unknown"
fi
echo "OS: $OS_ID"

# ── system packages ──────────────────────────────────────────────
echo "--- Installing system packages ---"
if [[ "$OS_ID" == "amzn" || "$OS_ID" == "rhel" || "$OS_ID" == "centos" ]]; then
    sudo yum update -y -q
    sudo yum install -y -q python3 python3-pip git gcc g++ make cmake
elif [[ "$OS_ID" == "ubuntu" || "$OS_ID" == "debian" ]]; then
    sudo apt-get update -qq
    sudo apt-get install -y -q python3 python3-pip git gcc g++ make cmake \
        fonts-noto-cjk
else
    echo "WARNING: unknown OS, skipping system package install"
fi

# ── Python venv ──────────────────────────────────────────────────
echo "--- Creating Python virtual environment ---"
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip -q

# ── llama-cpp-python (CPU build) ─────────────────────────────────
echo "--- Installing llama-cpp-python (CPU build) ---"
pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    -q

# GPU build (uncomment if using GPU instance e.g. g4dn, p3):
# CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python \
#     --upgrade --force-reinstall --no-cache-dir

# ── other dependencies ───────────────────────────────────────────
echo "--- Installing other Python packages ---"
pip install -r requirements.txt -q

# ── model download ───────────────────────────────────────────────
echo "--- Downloading model (~4.5 GB, may take several minutes) ---"
python download_model.py

# ── systemd service (optional, auto-start on boot) ───────────────
INSTALL_SERVICE="${INSTALL_SERVICE:-0}"
if [[ "$INSTALL_SERVICE" == "1" ]]; then
    echo "--- Installing systemd service ---"
    WORK_DIR="$(pwd)"
    VENV_PYTHON="$WORK_DIR/.venv/bin/python"
    SERVICE_FILE="/etc/systemd/system/llm-server.service"
    sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=Zero-Copy LLM Inference Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$WORK_DIR
ExecStart=$VENV_PYTHON server.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable llm-server
    sudo systemctl start llm-server
    echo "Service installed and started."
    echo "  Status : sudo systemctl status llm-server"
    echo "  Logs   : sudo journalctl -u llm-server -f"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Quickstart:"
echo "  source .venv/bin/activate"
echo ""
echo "  # Start server (foreground)"
echo "  python server.py"
echo ""
echo "  # In another terminal: send a prompt"
echo "  python client.py 'Q: What is the capital of France? A:'"
echo ""
echo "  # Run full benchmark"
echo "  python benchmark.py --n 3 --tokens 16"
echo ""
echo "  # Start server as daemon"
echo "  python server.py --daemon"
echo "  python server.py --status"
echo "  python server.py --stop-server"
echo ""
echo "  # Install as systemd service (auto-start on boot):"
echo "  INSTALL_SERVICE=1 bash setup.sh"
