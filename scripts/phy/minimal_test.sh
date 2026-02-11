SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../" && pwd)"
cd "${REPO_ROOT}"

CLUSTER_CONFIG=configs/cluster_minimal.json
TRACE=configs/trace_minimal.csv
CASE=minimal
POLICY=suika
NODE_ID=0
LOG_DIR=log

python -m Suika.cluster.executor \
	--node-id "${NODE_ID}" \
	--config "${CLUSTER_CONFIG}" \
	> "${LOG_DIR}/executor_node_${NODE_ID}.log" 2>&1 &

python scripts/phy/run_scheduler.py \
    --case "${CASE}" \
    --policy "${POLICY}" \
    --trace "${TRACE}" \
    --config "${CLUSTER_CONFIG}" \
    --ckpt "checkpoint_${POLICY}" 
    2>&1 | tee "${LOG_DIR}/${POLICY}_${CASE}.log"

echo "done"