#!/bin/bash
#SBATCH --job-name=so101-act-legalaspro
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --time=08:00:00

# ACT imitation learning on the legalaspro SO101 pick-and-place dataset.
#   sbatch scripts/train_act_legalaspro.sbatch
# The conda env is created on the first run if it does not exist yet.
set -euo pipefail

ENV_NAME="${ENV_NAME:-lerobot}"

# Flush prints straight to the slurm log instead of buffering them.
export PYTHONUNBUFFERED=1

source ~/miniconda3/etc/profile.d/conda.sh

# The dataset is lerobot format v3.0, which needs lerobot >= 0.4. It gets its own env so it does not
# clash with ManiSkill in the so101 env. ffmpeg is needed to decode the dataset videos.
if ! conda activate "${ENV_NAME}" 2>/dev/null; then
    echo "conda env '${ENV_NAME}' missing, creating it"
    conda create -y -n "${ENV_NAME}" python=3.10
    conda activate "${ENV_NAME}"
fi
if ! command -v lerobot-train >/dev/null; then
    echo "lerobot missing in '${ENV_NAME}', installing it"
    conda install -y -c conda-forge ffmpeg=7.1.1
    pip install "lerobot>=0.4"
fi
python -c "import torch, importlib.metadata as m; print('lerobot', m.version('lerobot'), '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

echo "host: $(hostname) | job: ${SLURM_JOB_ID:-n/a} | env: ${ENV_NAME}"
nvidia-smi

lerobot-train \
    --dataset.repo_id=legalaspro/so101-pick-and-place-cube-lerobot-30hz \
    --policy.type=act \
    --policy.device=cuda \
    --output_dir=/scratch/e1583535/projects/act_legalaspro \
    --steps=20000 \
    --batch_size=8 \
    --wandb.enable=false
