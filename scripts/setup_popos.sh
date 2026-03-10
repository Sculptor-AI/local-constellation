#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
LLAMA_DIR="${ROOT_DIR}/vendor/llama.cpp"

echo "[1/5] Installing base packages"
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  curl \
  git \
  pkg-config \
  python3 \
  python3-pip \
  python3-venv

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found. Install a working NVIDIA driver on the Pop!_OS host first." >&2
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  cat >&2 <<'EOF'
nvcc was not found.

Install a CUDA toolkit before building llama.cpp. On Pop!_OS this is commonly done via:
  - Ubuntu's nvidia-cuda-toolkit package, or
  - NVIDIA's official CUDA toolkit packages.

After CUDA is installed, rerun this script.
EOF
  exit 1
fi

echo "[2/5] Creating Python virtual environment"
python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --upgrade pip wheel
"${VENV_DIR}/bin/pip" install -e "${ROOT_DIR}"

echo "[3/5] Cloning or updating llama.cpp"
mkdir -p "${ROOT_DIR}/vendor"
if [[ -d "${LLAMA_DIR}/.git" ]]; then
  git -C "${LLAMA_DIR}" pull --ff-only
else
  git clone https://github.com/ggml-org/llama.cpp "${LLAMA_DIR}"
fi

echo "[4/5] Installing llama.cpp conversion dependencies"
"${VENV_DIR}/bin/pip" install -r "${LLAMA_DIR}/requirements/requirements-convert_hf_to_gguf.txt"

echo "[5/5] Building llama.cpp with CUDA"
cmake -S "${LLAMA_DIR}" -B "${LLAMA_DIR}/build" -DGGML_CUDA=ON
cmake --build "${LLAMA_DIR}/build" --config Release -j"$(nproc)"

cat <<EOF

Setup complete.

Next steps:
  1. Copy ${ROOT_DIR}/.env.example to ${ROOT_DIR}/.env and adjust any paths.
  2. Start the manager API:
     ${VENV_DIR}/bin/python -m local_constellation
  3. Pull a model via POST /models/pull, then start llama-server via POST /server/start.
EOF
