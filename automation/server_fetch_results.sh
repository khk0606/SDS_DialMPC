#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./automation/server_fetch_results.sh <server> <remote_repo_dir> <sweep_name> [local_dir]
#
# Example:
#   ./automation/server_fetch_results.sh ubuntu@1.2.3.4 /home/ubuntu/SDS_DialMPC sweep_2026-02-18 ./runs

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <server> <remote_repo_dir> <sweep_name> [local_dir]" >&2
  exit 1
fi

SERVER="$1"
REMOTE_REPO="$2"
SWEEP_NAME="$3"
LOCAL_DIR="${4:-./runs}"

mkdir -p "${LOCAL_DIR}"
rsync -azP "${SERVER}:${REMOTE_REPO}/runs/${SWEEP_NAME}/" "${LOCAL_DIR}/${SWEEP_NAME}/"
rsync -azP "${SERVER}:${REMOTE_REPO}/logs/" "${LOCAL_DIR}/../logs/"
echo "[OK] fetched ${SWEEP_NAME} into ${LOCAL_DIR}/${SWEEP_NAME}"
