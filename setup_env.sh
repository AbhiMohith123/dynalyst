#!/bin/bash
# One-time setup for a new machine (the GPU server or a laptop).
# Run from the repository root:   bash setup_env.sh
# (tmux/screen recommended: installing torch/jax and building AUTO-07p takes a while.)
set -e

ENV_NAME="${ENV_NAME:-dynalyst}"
PY_VERSION="3.11"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# AUTO-07p and jaxkineticmodel are shipped as zip files in vendor/ (the web upload limits the number of files)
for z in auto-07p jaxkineticmodel; do [ -d vendor/$z ] || { command -v unzip >/dev/null && unzip -q vendor/$z.zip -d vendor || python -m zipfile -e vendor/$z.zip vendor; }; done

echo "== Checking GPU =="
if nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
  GPU_AVAILABLE=1
else
  echo "No usable GPU from this shell -- continuing on CPU (embeddings and"
  echo "jaxkineticmodel simulations still work, just slower)."
  GPU_AVAILABLE=0
fi

echo "== Creating conda env: ${ENV_NAME} =="
conda create -n "${ENV_NAME}" python="${PY_VERSION}" -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "== Installing PyTorch (used only by the RAG embedding/rerank models) =="
# transformers needs torch >= 2.6 (with an older torch it switches torch off and the RAG
# tools fail with "NameError: name 'nn' is not defined"). The torch wheels on PyPI already
# include CUDA on Linux; the old cu121 index stops at torch 2.5.1, so it is not used.
pip install "torch>=2.6"

echo "== Installing JAX (jaxkineticmodel pins jax 0.4.35) =="
# CPU JAX by default: the agent's simulations are small, and the CUDA build of jax 0.4.35
# predates recent GPUs (e.g. Blackwell). JAX_GPU=1 bash setup_env.sh installs the CUDA build.
if [ "${JAX_GPU:-0}" -eq 1 ] && [ "$GPU_AVAILABLE" -eq 1 ]; then
  pip install "jax[cuda12]==0.4.35"
else
  pip install "jax==0.4.35"
  # jax 0.4.35 crashes at import when an nvidia/cuda_nvcc folder without the package is left
  # in the environment (e.g. by another CUDA package); with CUDA_ROOT set it skips that lookup.
  conda env config vars set CUDA_ROOT="$CONDA_PREFIX" >/dev/null   # this env only, from now on
  export CUDA_ROOT="$CONDA_PREFIX"                                  # and for the rest of this script
fi

echo "== Installing jaxkineticmodel from vendor/jaxkineticmodel =="
pip install -e vendor/jaxkineticmodel

echo "== Installing the agents, RAG stack and analysis dependencies =="
pip install -r requirements.txt

echo "== Installing crnt4sbml (reaction network theory) =="
# Its pinned dependencies (numpy 1.16, sympy 1.4, ...) cannot coexist with
# jax; it runs on current versions (dynalyst aliases the few removed names).
pip install --no-deps "crnt4sbml==0.0.15"

echo "== Installing this project (package: dynalyst) =="
pip install -e .

echo "== Building AUTO-07p from vendor/auto-07p =="
# Needs gfortran, e.g. on Debian/Ubuntu `sudo apt-get install -y gfortran`, on macOS
# `brew install gcc`, or without root `conda install -c conda-forge gfortran`. The
# library is built into vendor/auto-07p/lib/<platform>/ and checked with a small
# continuation -- see scripts/build_auto.py.
python scripts/build_auto.py

echo "== Sanity checks =="
python -c "import jax; print('JAX devices:', jax.devices())"
python -c "import torch; print('Torch CUDA available:', torch.cuda.is_available())"
python -c "from jaxkineticmodel.load_sbml.sbml_model import SBMLModel; print('jaxkineticmodel import OK')"
python -c "from dynalyst.auto_env import ensure_auto_env; ensure_auto_env(); import auto; print('AUTO-07p Python interface OK')"
python -c "import dynalyst.agents.unified_agent; print('agents import OK')"

echo ""
echo "== Setup complete =="
echo "Run the tests (AUTO integration included):   python -m pytest -q"
echo "Set your API key before using an agent:"
echo "  export DYNALYST_MODEL=openai:gpt-4o-mini  OPENAI_API_KEY=your-key   (other providers: README, step 6)"
echo "The knowledge-base index is built separately -- see README, 'Knowledge base'."
