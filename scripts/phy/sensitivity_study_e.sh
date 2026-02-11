# Set these env vars if you wish to adopt other setups
# See aux_run.sh for usage details
set -euo pipefail
set -a

# SUIKA_CONDA_ENV=Suika
SUIKA_TRACE=trace_64_final.csv
SUIKA_CLUSTER_CONFIG=configs/cluster64.json
SUIKA_TRACE=configs/trace_64_final.csv
SUIKA_INTENSITY=1.0
SUIKA_NUM_NODES=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SUIKA_CASE=clusterA
bash aux_run.sh sjf

SUIKA_CASE=sjf
SUIKA_POLICY_BASE=sjf
bash aux_run.sh suika