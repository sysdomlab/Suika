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

SUIKA_CASE=lbd0.25
SUIKA_POLICY_TH_EXP=0.25
bash aux_run.sh suika

SUIKA_CASE=lbd0.50
SUIKA_POLICY_TH_EXP=0.50
bash aux_run.sh suika


SUIKA_CASE=lbd0.75
SUIKA_POLICY_TH_EXP=0.75
bash aux_run.sh suika

# SKIP, gather from clusterA (main e2e)
# SUIKA_CASE=lbd1.00
# SUIKA_POLICY_TH_EXP=1.00
# bash aux_run.sh suika

SUIKA_CASE=lbd1.25
SUIKA_POLICY_TH_EXP=1.25
bash aux_run.sh suika

SUIKA_CASE=lbd1.50
SUIKA_POLICY_TH_EXP=1.50
bash aux_run.sh suika

SUIKA_CASE=lbd1.75
SUIKA_POLICY_TH_EXP=1.75
bash aux_run.sh suika

SUIKA_CASE=lbd2.00
SUIKA_POLICY_TH_EXP=2.00
bash aux_run.sh suika