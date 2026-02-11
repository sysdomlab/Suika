# Set these env vars if you wish to adopt other setups
# See aux_run.sh for usage details
set -euo pipefail
set -a

# SUIKA_CONDA_ENV=Suika
SUIKA_CLUSTER_CONFIG=configs/cluster64.json
SUIKA_INTENSITY=1.0
SUIKA_NUM_NODES=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SUIKA_CASE=philly2
SUIKA_TRACE=configs/trace_64_philly2.csv
bash aux_run.sh suika
bash aux_run.sh sia
bash aux_run.sh rubick
bash aux_run.sh fcfs

SUIKA_CASE=helios_venus
SUIKA_TRACE=configs/trace_64_helios_venus.csv
bash aux_run.sh suika
bash aux_run.sh sia
bash aux_run.sh rubick
bash aux_run.sh fcfs

SUIKA_CASE=helios_uranus
SUIKA_TRACE=configs/trace_64_helios_uranus.csv
bash aux_run.sh suika
bash aux_run.sh sia
bash aux_run.sh rubick
bash aux_run.sh fcfs