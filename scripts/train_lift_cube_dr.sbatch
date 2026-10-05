#!/bin/bash
#SBATCH --job-name=so101-grasp-cube-dr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
# xgpi7: GPU camera rendering (Vulkan) hangs in create_camera_group on this node
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --time=04:00:00

# RGB PPO training of SO101GraspCube-v1 (grasp + lift cube, return to rest pose) with domain randomization.
#
# Submit from the repo root:
#   sbatch scripts/train_lift_cube_dr.sbatch
# The conda env is created automatically (scripts/setup_env.sh) if it does not exist yet.
#
# Overridable via environment variables, e.g.  SEED=2 NUM_ENVS=2048 TRACK=1 sbatch scripts/train_lift_cube_dr.sbatch
#   ENV_NAME      conda env name                         (default: so101)
#   ENV_ID        ManiSkill env id                       (default: SO101GraspCube-v1)
#   ENV_CONFIG    env kwargs json                        (default: env_config.json)
#   SEED          random seed                            (default: 3)
#   NUM_ENVS      parallel training envs                 (default: 2048, fits easily in 96GB)
#   TOTAL_STEPS   total env steps                        (default: 100_000_000)
#   TRACK         1 = log to Weights & Biases            (default: 0)
#   WANDB_ENTITY  W&B entity, required when TRACK=1 (the code default "uqyluo" belongs to the upstream author)
#   WANDB_PROJECT W&B project                            (default: SO101-ManiSkill)
set -euo pipefail

ENV_NAME="${ENV_NAME:-so101}"
ENV_ID="${ENV_ID:-SO101GraspCube-v1}"
ENV_CONFIG="${ENV_CONFIG:-env_config.json}"
SEED="${SEED:-1000}"
NUM_ENVS="${NUM_ENVS:-1024}"
TOTAL_STEPS="${TOTAL_STEPS:-4_000_000}"
TRACK="${TRACK:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-SO101-ManiSkill}"
EXP_NAME="${EXP_NAME:-${SLURM_JOB_NAME:-ppo-${ENV_ID}-rgb}-s${SEED}-${SLURM_JOB_ID:-local}}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Make lerobot_sim2real importable from the repo root regardless of how pip's editable install behaves.
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
# Flush prints straight to the slurm log instead of buffering them.
export PYTHONUNBUFFERED=1

source ~/miniconda3/etc/profile.d/conda.sh

# Create the env on first use. The lock stops concurrent jobs from installing into it at the same time;
# the import check also catches an env left half-installed by an earlier failed setup.
env_ready() {
    conda run -n "${ENV_NAME}" python -c "import torch, mani_skill, lerobot_sim2real" >/dev/null 2>&1
}
if ! env_ready; then
    mkdir -p ~/.cache
    (
        flock 9
        if ! env_ready; then
            echo "conda env '${ENV_NAME}' missing or incomplete, running scripts/setup_env.sh"
            ENV_NAME="${ENV_NAME}" bash scripts/setup_env.sh
        fi
    ) 9>"${HOME}/.cache/setup_env_${ENV_NAME}.lock"
fi
conda activate "${ENV_NAME}"

echo "host: $(hostname) | job: ${SLURM_JOB_ID:-n/a} | exp: ${EXP_NAME}"
echo "python: $(command -v python) | repo: $(pwd)"
python -c "import lerobot_sim2real, mani_skill; print('lerobot_sim2real from', lerobot_sim2real.__path__)"
nvidia-smi

# The env config points at a greenscreen image; fail early instead of crashing after env setup.
GREENSCREEN=$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('greenscreen_overlay_path') or '')" "${ENV_CONFIG}")
if [[ -n "${GREENSCREEN}" && ! -f "${GREENSCREEN}" ]]; then
    echo "ERROR: ${ENV_CONFIG} references greenscreen '${GREENSCREEN}' which does not exist." >&2
    echo "       Copy it to the cluster, or set greenscreen_overlay_path to null to train without it." >&2
    exit 1
fi

# GPU camera rendering (Vulkan + CUDA interop) hangs instead of erroring on some nodes. Check it on a
# tiny env first so a bad node fails in minutes rather than burning the whole allocation.
echo "vulkan ICD files: $(ls /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d/ 2>/dev/null | tr '\n' ' ') | VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-unset}"
if ! timeout 180 python - "${ENV_ID}" <<'EOF'
import sys, time
import gymnasium as gym
import mani_skill.envs  # noqa: F401
t0 = time.time()
env = gym.make(sys.argv[1], num_envs=2, obs_mode="rgb+segmentation", sim_backend="physx_cuda")
env.reset(seed=0)
env.close()
print(f"render check passed in {time.time() - t0:.1f}s")
EOF
then
    echo "ERROR: GPU rendering check failed or hung on $(hostname). Vulkan is likely broken on this node." >&2
    echo "       Resubmit with: sbatch --exclude=$(hostname -s) scripts/train_lift_cube_dr.sbatch" >&2
    exit 1
fi

TRACK_ARGS=()
if [[ "${TRACK}" == "1" ]]; then
    : "${WANDB_ENTITY:?set WANDB_ENTITY when TRACK=1}"
    TRACK_ARGS=(--ppo.track --ppo.wandb_project_name "${WANDB_PROJECT}" --ppo.wandb_entity "${WANDB_ENTITY}")
fi

# Hyperparameters follow the tuned command in README.md (num_envs bumped for the H100).
python lerobot_sim2real/scripts/train_ppo_rgb.py \
    --env-id="${ENV_ID}" --env-kwargs-json-path="${ENV_CONFIG}" \
    --ppo.seed="${SEED}" \
    --ppo.num_envs="${NUM_ENVS}" --ppo.num-steps=16 --ppo.update_epochs=8 --ppo.num_minibatches=32 \
    --ppo.total_timesteps="${TOTAL_STEPS}" --ppo.gamma=0.9 \
    --ppo.num_eval_envs=16 --ppo.num-eval-steps=64 --ppo.no-partial-reset \
    --ppo.exp-name="${EXP_NAME}" \
    ${TRACK_ARGS[@]+"${TRACK_ARGS[@]}"}
