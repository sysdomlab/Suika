set -euo pipefail
set -a
SUIKA_TESTBED=cluster-d # FOR SIMU
SUIKA_SIMULATION=True
SUIKA_SIMU_SCHEDULE_INTERVAL=100000000000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../"
mkdir -p log

# list=(64 128 256 512 1024)
list=(64 128)
for i in "${list[@]}"; do
    CLUSTER_CONFIG="configs/cluster${i}.json"
    TRACE="configs/trace_${i}_so.csv"
    CASE="so-${i}"

    for POLICY in suika sia rubick; do
        # to parallelize, add & to run in background
        python scripts/simu/run_simulation.py --case "${CASE}" --policy "${POLICY}" --trace "${TRACE}" --config "${CLUSTER_CONFIG}" 2>&1 | tee "log/${CASE}-${POLICY}-main.log" 
    done
done