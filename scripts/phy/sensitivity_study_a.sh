# Set these env vars if you wish to adopt other setups
# See aux_run.sh for usage details
set -euo pipefail
set -a

# SUIKA_CONDA_ENV=Suika
SUIKA_TRACE=configs/trace_64_final.csv
SUIKA_INTENSITY=1.0
SUIKA_NUM_NODES=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

printf '%s\n' 'uncomment `tasks/task.sh: line 20` for cluster B test; press enter to proceed'
read -r -p ""

SUIKA_CASE=clusterB
SUIKA_TESTBED=cluster-b
SUIKA_CLUSTER_CONFIG=configs/cluster64.json
bash aux_run.sh suika
bash aux_run.sh sia
bash aux_run.sh rubick
bash aux_run.sh fcfs

printf '%s\n' 'uncomment `tasks/task.sh: line 23` for cluster C test; press enter to proceed'
read -r -p ""

SUIKA_CASE=clusterC
SUIKA_TESTBED=cluster-c
SUIKA_CLUSTER_CONFIG=configs/cluster64_no_ib.json
bash aux_run.sh suika
bash aux_run.sh sia
bash aux_run.sh rubick
bash aux_run.sh fcfs

printf '%s\n' 'Remeber to comment out `tasks/task.sh: line 20/23` before running other tests; press enter to proceed'
read -r -p ""