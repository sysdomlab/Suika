#!/usr/bin/env bash

# Entry-point script for experiments (main_e2e.sh and sensitivity_study_{a~f}.sh)
# Usage: aux_run.sh <policy> [case] [trace] [intensity] [num_nodes] [conda_env]
# - policy: scheduling policy name (e.g., suika, fcfs, sjf) [REQUIRED positional]
# - case: case name label for logs and outputs [defaults to $SUIKA_CASE]
# - trace: path to trace CSV [defaults to $SUIKA_TRACE]
# - intensity: workload intensity scalar [defaults to $SUIKA_INTENSITY or 1.0]
# - num_nodes: number of nodes to launch remote executors on; if 0, skip [defaults to $SUIKA_NUM_NODES or 0]
# - conda_env: conda environment name to activate; if omitted, falls back to $SUIKA_CONDA_ENV if set
SUIKA_WAIT_FOR=1

usage() {
        echo "Usage: $(basename "$0") <policy> [case] [trace] [intensity] [num_nodes] [conda_env]" >&2
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
        usage
        exit 0
fi

if [[ $# -lt 1 ]]; then
        usage
        exit 1
fi

# Set CWD to repo root (script_dir/../../)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../" && pwd)"
cd "${REPO_ROOT}"

POLICY="$1"
CASE="${2:-${SUIKA_CASE:-}}"
TRACE="${3:-${SUIKA_TRACE:-}}"
INTENSITY="${4:-${SUIKA_INTENSITY:-1.0}}"
NUM_NODES="${5:-${SUIKA_NUM_NODES:-0}}"
CONDA_ENV_INPUT="${6:-}"

# Determine conda env: use last optional arg if provided, else env var
CONDA_ENV="${CONDA_ENV_INPUT:-${SUIKA_CONDA_ENV:-}}"
if [[ -n "${CONDA_ENV}" ]]; then
    if [[ -f "$HOME/.bashrc" ]]; then
        source "$HOME/.bashrc"
    fi
    conda activate "${CONDA_ENV}" || echo "Warning: failed to activate conda env '${CONDA_ENV}'."
fi

# Ensure required CASE and TRACE are set (from args or env)
missing=()
[[ -z "${CASE}" ]] && missing+=("case (arg2 or SUIKA_CASE)")
[[ -z "${TRACE}" ]] && missing+=("trace (arg3 or SUIKA_TRACE)")
if [[ ${#missing[@]} -gt 0 ]]; then
        echo "Error: missing required values: ${missing[*]}" >&2
        usage
        exit 1
fi

CLUSTER_CONFIG="${SUIKA_CLUSTER_CONFIG:-cluster_config/cluster64.json}"
WORKSPACE="${SUIKA_WORKSPACE:-${REPO_ROOT}}"

# Start executors on remote if requested; otherwise, print hint for local start
if [[ "${NUM_NODES}" -gt 0 ]]; then
        # echo "Starting remote executors on ${NUM_NODES} nodes..."
        bash "${SCRIPT_DIR}/aux_executor_remote.sh" "${NUM_NODES}" "${CLUSTER_CONFIG}" "${CONDA_ENV}" "${WORKSPACE}" > /dev/null 2>&1 &
else
        :
        # echo "num_nodes=${NUM_NODES}. Remote executors not started. You can start a local executor with:"
        # echo "  ${SCRIPT_DIR}/aux_executor_local.sh <node_id> ${CLUSTER_CONFIG} ${CONDA_ENV}"
fi

sleep 2

LOG_DIR="log/log_${POLICY}_${CASE}"
mkdir -p "${LOG_DIR}"

export SUIKA_SIMULATION=False
# echo "Starting scheduler: policy=${POLICY}, case=${CASE}, trace=${TRACE}, intensity=${INTENSITY}"
python "${SCRIPT_DIR}/run_scheduler.py" \
        --policy "${POLICY}" \
        --trace "${TRACE}" \
        --config "${CLUSTER_CONFIG}" \
        --ckpt "checkpoint_${POLICY}" \
        --intensity "${INTENSITY}" \
        --casename "${CASE}" \
        2>&1 | tee "${LOG_DIR}/${POLICY}_${CASE}.log" &

PID=$!
echo "${PID}" > /tmp/aux_run.sh.pid
# echo "Scheduler started (pid=${PID}). Logs: ${LOG_DIR}/${POLICY}_${CASE}.log"
# you may use this PID in another script, to add other functionality like "wait with timeout".

SUIKA_WAIT_FOR=1
if [[ "${SUIKA_WAIT_FOR}" -eq 1 ]]; then
    # echo "Waiting for scheduler process (pid=${PID}) to finish..."
    wait "${PID}"
    # echo "Scheduler process finished."
fi