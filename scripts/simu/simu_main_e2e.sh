set -euo pipefail
set -a
SUIKA_TESTBED=cluster-a
SUIKA_SIMULATION=True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../"
mkdir -p log

CLUSTER_CONFIG="configs/cluster64.json"
TRACE="configs/trace_64_final.csv"
CASE="simu-main-e2e"
for POLICY in suika sia rubick fcfs; do
    # to parallelize, add & to run in background
    python scripts/simu/run_simulation.py --case "${CASE}" --policy "${POLICY}" --trace "${TRACE}" --config "${CLUSTER_CONFIG}" 2>&1 | tee "log/${CASE}-${POLICY}-main.log" 
done

python scripts/simu/compare_phy_simu_main_e2e.py