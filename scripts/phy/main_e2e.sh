# Set these env vars if you wish to adopt other setups
# See aux_run.sh for usage details

set -euo pipefail
set -a
# SUIKA_CONDA_ENV=Suika
SUIKA_CASE=clusterA
SUIKA_TRACE=configs/trace_64_final.csv
SUIKA_CLUSTER_CONFIG=configs/cluster64.json
SUIKA_INTENSITY=1.0
SUIKA_NUM_NODES=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

bash aux_run.sh suika
bash aux_run.sh sia
bash aux_run.sh rubick
bash aux_run.sh fcfs
bash aux_run.sh metis
bash aux_run.sh dp
