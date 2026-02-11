# Set these env vars if you wish to adopt other setups
# See aux_run.sh for usage details
set -euo pipefail
set -a

# SUIKA_CONDA_ENV=Suika
SUIKA_CLUSTER_CONFIG=configs/cluster64.json
SUIKA_TRACE=configs/trace_64_final.csv
SUIKA_NUM_NODES=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SUIKA_CASE=int0.75
SUIKA_INTENSITY=0.75
bash aux_run.sh suika
bash aux_run.sh sia
bash aux_run.sh rubick
bash aux_run.sh fcfs

SUIKA_CASE=int1.25
SUIKA_INTENSITY=1.25
bash aux_run.sh suika
bash aux_run.sh sia
bash aux_run.sh rubick
bash aux_run.sh fcfs

SUIKA_CASE=int1.5
SUIKA_INTENSITY=1.5
bash aux_run.sh suika
bash aux_run.sh sia
bash aux_run.sh rubick
bash aux_run.sh fcfs

SUIKA_CASE=int2.0
SUIKA_INTENSITY=2.0
bash aux_run.sh suika
bash aux_run.sh sia
bash aux_run.sh rubick
bash aux_run.sh fcfs

