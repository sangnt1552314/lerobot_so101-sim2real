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
#   ENV_NAME=lerobot sbatch scripts/train_act_legalaspro.sbatch   # use another conda env
set -euo pipefail

ENV_NAME="${ENV_NAME:-so101}"

# Flush prints straight to the slurm log instead of buffering them.
export PYTHONUNBUFFERED=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

# scripts/setup_env.sh does not install lerobot, so fail early if it is missing.
command -v lerobot-train >/dev/null || {
    echo "ERROR: lerobot-train not found in conda env '${ENV_NAME}'; run: pip install lerobot==0.3.4" >&2
    exit 1
}

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
