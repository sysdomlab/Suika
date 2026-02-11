#!/usr/bin/env bash
set -euo pipefail

# Launch the executor locally on a worker node.
# Usage: aux_executor_local.sh <node_id> <cluster_config> [conda_env] [python_bin]
# - node_id: integer node id that exists in the cluster config JSON
# - cluster_config: path to cluster config JSON
# - conda_env (optional): name of conda env to activate (or use $SUIKA_CONDA_ENV)

usage() {
	echo "Usage: $(basename "$0") <node_id> <cluster_config> [conda_env]" >&2
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
	usage
	exit 0
fi

if [[ $# -lt 2 ]]; then
	usage
	exit 1
fi

NODE_ID="$1"
CLUSTER_CONFIG="$2"
CONDA_ENV="${3:-${SUIKA_CONDA_ENV:-}}"

# Set CWD to repo root (script_dir/../../)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../" && pwd)"
cd "${REPO_ROOT}"

# Optional conda activation
if [[ -n "${CONDA_ENV}" ]]; then
	# Load bashrc to ensure 'conda' is on PATH
	if [[ -f "$HOME/.bashrc" ]]; then
		source "$HOME/.bashrc"
	fi
	conda activate "${CONDA_ENV}" || {
		echo "Warning: failed to activate conda env '${CONDA_ENV}'. Continuing without activation." >&2
	}
fi

LOG_DIR="${SUIKA_LOG_DIR:-log}"
mkdir -p "${LOG_DIR}"

# Kill any existing executor processes
bash scripts/phy/aux_kill.sh

# Launch the executor in the background, redirecting stdout and stderr to log files.

nohup python -m Suika.cluster.executor \
	--node-id "${NODE_ID}" \
	--config "${CLUSTER_CONFIG}" \
	> "${LOG_DIR}/executor_node_${NODE_ID}.log" 2>&1 &

PID=$!
echo "${PID}" > "/tmp/executor_${NODE_ID}.pid"
echo "Started local executor for node ${NODE_ID} (pid=${PID}). Logs: ${LOG_DIR}/executor_node_${NODE_ID}.log"


