source /opt/conda/bin/activate base
source ~/.bashrc

./aux_executor_remote.sh > /dev/null 2>&1

sleep 2

policy=$1
case="micro"
intensity="1"
trace="trace_small.csv"
export SUIKA_LOG_DIR=log_${policy}_${case}

export SUIKA_POLICY_TH_EXP=100
export METIS_TIME_BUDGET=180

# export SUIKA_TESTBED="cluster-b" # also check HALF NICs DISABLED in task.sh
# export SUIKA_TESTBED="cluster-c" # also check IB DISABLED in task.sh

python run_micro_scopic.py --policy ${policy} --trace ${trace} --ckpt checkpoint_${policy} --intensity ${intensity} --casename ${case} > ${policy}_${case}.log 2>&1