set -euo pipefail
set -a
SUIKA_TESTBED=cluster-d # FOR SIMU
SUIKA_SIMULATION=True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../"
mkdir -p log

CLUSTER_CONFIG="configs/cluster1024.json"
list=(1 2 3 4)
for i in "${list[@]}"; do
    TRACE="configs/trace_1024_${i}.csv"
    CASE="simu-trace${i}"
    for POLICY in suika sia rubick fcfs; do
        # to parallelize, add & to run in background
        python scripts/simu/run_simulation.py --case "${CASE}" --policy "${POLICY}" --trace "${TRACE}" --config "${CLUSTER_CONFIG}" 2>&1 | tee "log/${CASE}-${POLICY}-main.log" 
    done
done