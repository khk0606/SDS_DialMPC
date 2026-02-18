#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./automation/server_submit_sweep.sh <server> <remote_repo_dir> <python_bin> [trials]
#
# Example:
#   ./automation/server_submit_sweep.sh ubuntu@1.2.3.4 /home/ubuntu/SDS_DialMPC \
#     /home/ubuntu/miniconda3/envs/sds/bin/python 200

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <server> <remote_repo_dir> <python_bin> [trials]" >&2
  exit 1
fi

SERVER="$1"
REMOTE_REPO="$2"
PYTHON_BIN="$3"
TRIALS="${4:-200}"

REMOTE_CMD="
set -euo pipefail
cd '${REMOTE_REPO}'
mkdir -p logs runs
git pull --ff-only
'${PYTHON_BIN}' -m pip install -q pyyaml || true
nohup '${PYTHON_BIN}' automation/run_sweep.py \
  --config dial-mpc/dial_mpc/examples/sds_gallop_sim.yaml \
  --space automation/sweep_space.yaml \
  --out runs/sweep_\$(date +%F) \
  --trials ${TRIALS} \
  --topk 20 \
  --timeout-sec 1800 \
  --resume \
  > logs/sweep_\$(date +%F).log 2>&1 &
echo '[OK] sweep submitted on server.'
"

ssh "${SERVER}" "${REMOTE_CMD}"
