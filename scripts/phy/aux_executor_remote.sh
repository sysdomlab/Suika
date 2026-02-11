#!/usr/bin/env bash
set -euo pipefail

# Launch executors on remote worker nodes via SSH.
# Assumes SSH alias pattern: H100-worker-0{i} and passwordless access.
# Usage: aux_executor_remote.sh <num_nodes> <cluster_config> [conda_env] [workspace]
# - num_nodes: total number of worker nodes
# - cluster_config: path to cluster config JSON (accessible on remote nodes)
# - conda_env (optional): name of conda env to activate on remote
# - workspace (optional): path to workspace on remote nodes (default: $SUIKA_WORKSPACE or repo root)

usage() {
	echo "Usage: $(basename "$0") <num_nodes> <cluster_config> [conda_env] [workspace]" >&2
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
	usage
	exit 0
fi

if [[ $# -lt 2 ]]; then
	usage
	exit 1
fi

NUM_NODES="$1"
CLUSTER_CONFIG="$2"
CONDA_ENV="${3:-${SUIKA_CONDA_ENV:-}}"

# Determine repo root and default workspace
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../" && pwd)"
WORKSPACE="${4:-${SUIKA_WORKSPACE:-${REPO_ROOT}}}"

for (( i=0; i<NUM_NODES; i++ )); do
	
	HOST="H100-worker-0${i}"

	echo "Starting executor on ${HOST} (node_id=${i})..."

	# Compose remote command: call the local helper on remote host
	REMOTE_CMD="set -euo pipefail; cd \"${WORKSPACE}\"; \
bash scripts/phy/aux_executor_local.sh ${i} \"${CLUSTER_CONFIG}\" \"${CONDA_ENV}\""

	# Execute over SSH
	ssh -o BatchMode=yes -o ConnectTimeout=10 "${HOST}" "bash -lc '${REMOTE_CMD}'" || {
		echo "Error: failed to start executor on ${HOST}." >&2
		exit 1
	}
	echo "Executor started on ${HOST}."
done

echo "All remote executors started on ${NUM_NODES} nodes."