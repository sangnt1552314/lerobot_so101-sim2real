#!/bin/bash
# Create the conda environment used for simulation training on the cluster.
#
# Usage (run from the repo root, on a Linux node; a GPU node is needed for the sanity check):
#   bash scripts/setup_env.sh
#
# Overridable via environment variables:
#   ENV_NAME     conda env name                       (default: so101)
#   PY_VERSION   python version                       (default: 3.11)
#   TORCH_INDEX  PyTorch wheel index (CUDA build)     (default: cu124)
set -euo pipefail

ENV_NAME="${ENV_NAME:-so101}"
PY_VERSION="${PY_VERSION:-3.11}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source ~/miniconda3/etc/profile.d/conda.sh

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[setup] conda env '${ENV_NAME}' already exists, reusing it"
else
    echo "[setup] creating conda env '${ENV_NAME}' (python ${PY_VERSION})"
    conda create -y -n "${ENV_NAME}" "python=${PY_VERSION}"
fi
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip

# torch first so ManiSkill does not pull a mismatched CUDA build
pip install torch torchvision --index-url "${TORCH_INDEX}"

# ManiSkill (local fork with the SO101 env) + this package
pip install -e "${REPO_DIR}/ManiSkill"
pip install -e "${REPO_DIR}"

# lerobot is only needed for real-robot deployment, not for sim training.
# The lerobot/ folder in this repo is an empty submodule pointer, so install from PyPI if you need it:
#   pip install "lerobot==0.3.4"

echo "[setup] verifying install"
python - <<'EOF'
import torch, mani_skill, sapien
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("mani_skill", mani_skill.__version__, "| sapien", sapien.__version__)
EOF

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[setup] GPU found, creating SO101GraspCube-v1 as a smoke test"
    python - <<'EOF'
import gymnasium as gym
import mani_skill.envs  # noqa: F401
env = gym.make("SO101GraspCube-v1", num_envs=2, obs_mode="rgb+segmentation", sim_backend="physx_cuda")
env.reset(seed=0)
env.close()
print("SO101GraspCube-v1 OK")
EOF
else
    echo "[setup] no GPU on this node, skipping env smoke test (run 'python -m mani_skill.examples.demo_random_action' on a GPU node)"
fi

echo "[setup] done. Activate with: source ~/miniconda3/etc/profile.d/conda.sh && conda activate ${ENV_NAME}"
