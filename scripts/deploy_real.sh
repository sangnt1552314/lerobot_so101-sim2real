#!/bin/bash
# Deploy a trained RGB PPO checkpoint on the real SO101 arm.
#   bash scripts/deploy_real.sh                         # run the policy
#   bash scripts/deploy_real.sh --no-continuous-eval    # press enter before every step
#   bash scripts/deploy_real.sh --debug                 # show sim/real camera overlay
# Extra arguments are passed to eval_ppo_rgb.py. Ctrl+C returns the arm to rest and exits.
set -euo pipefail

CHECKPOINT="runs/so101-grasp-cube-dr-s1000-911596/final_ckpt.pt"
ROBOT_PORT="/dev/tty.usbmodem5C821064861"
ROBOT_ID="home_follower"

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

python lerobot_sim2real/scripts/eval_ppo_rgb.py \
    --env-id SO101GraspCube-v1 --env-kwargs-json-path env_config.json \
    --checkpoint "${CHECKPOINT}" \
    --robot-port "${ROBOT_PORT}" --robot-id "${ROBOT_ID}" \
    --control-freq 15 \
    "$@"
