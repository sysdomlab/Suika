
from Suika.cluster.resource import (
  default_acc_type, 
  default_acc_TFLOPS, 
  default_acc_memory, 
  default_intra_bandwidth, 
  default_inter_bandwidth,
  default_n_acc_per_node,
  default_n_node,
)

k_acc_type = default_acc_type
k_acc_TFLOPS = default_acc_TFLOPS
k_acc_memory = default_acc_memory
k_intra_bandwidth = default_intra_bandwidth
k_inter_bandwidth = default_inter_bandwidth

# resource manager to create estimator that performs feasibility test
# only 'memory' field is used
def get_rubick_rm():
  from Suika.cluster.resource import get_default_rm
  return get_default_rm()
  
k_n_acc_per_node = default_n_acc_per_node
k_num_node = default_n_node
k_max_n_acc = k_n_acc_per_node * k_num_node

# policy related 
k_queuing_penalty_threshold = 200000 / 1000 # in second
# k_queuing_penalty_threshold = 0 / 1000 # in second
k_reconfig_penalty_threshold = 0.97

perf_curve_padding = k_n_acc_per_node
perf_curve_min = -perf_curve_padding
perf_curve_max = k_max_n_acc+ perf_curve_padding

import os
# all hyperparameters below are tuned in favor of Rubick's performance.

# whether to patch rubick to use non local job slope; see suika_interface _get_job_slope.
# the original rubick artifacts use local job slope, but it acts poorly when many jobs are multi-gpu, as +-1 delta n_gpu is likely leading to 0 slope.
patch_rubick_non_local_job_slope = os.getenv("RUBICK_NON_LOCAL_JOB_SLOPE", "True").lower() in ["true", "yes", "y", "1"] 

# whether to patch rubick to preempt jobs with high reconfig penality when there are jobs queuing too long.
patch_rubick_queuing_preempt = os.getenv("RUBICK_QUEUING_PREEMPT", "True").lower() in ["true", "yes", "y", "1"] 

# whether to treat all jobs as best effort or guaranteed
patch_rubick_best_effort = os.getenv("RUBICK_BEST_EFFORT", "False").lower() in ["true", "yes", "y", "1"] 

# whether to half NVLINK bandwidth to account for underutilization
#   Rubick favors high-TP-degree plans when scaling up which are, in fact, not optimal
#   Though we have discounted the effective NVLINK bandwidth to 300 GB/s unilateral, since our NVLINK is about twice the speed as Rubick's original testbed, 
#     the underutilization issue is still notable
#   Further halfing NVLINK bandwidth (for Rubick's performance model, not physical) yields better estimation.
patch_rubick_half_nvlink_bandwidth = os.getenv("RUBICK_HALF_NVLINK_BANDWIDTH", "True").lower() in ["true", "yes", "y", "1"]

is_simulation = os.getenv("SUIKA_SIMULATION", "False").lower() in ["true", "yes", "y", "1"]

patch_rubick_tp_commu = os.getenv("RUBICK_TP_COMMU", str(is_simulation)).lower() in ["true", "yes", "y", "1"]

if patch_rubick_half_nvlink_bandwidth and not is_simulation:
  k_intra_bandwidth /= 2

from Suika.external.Rubick.utils import rubick_logger
rubick_logger.info(f"RUBICK_NON_LOCAL_JOB_SLOPE {patch_rubick_non_local_job_slope}")
rubick_logger.info(f"RUBICK_QUEUING_PREEMPT {patch_rubick_queuing_preempt}")
rubick_logger.info(f"RUBICK_BEST_EFFORT {patch_rubick_best_effort}")
rubick_logger.info(f"RUBICK_HALF_NVLINK_BANDWIDTH {patch_rubick_half_nvlink_bandwidth} / intra bw {k_intra_bandwidth}")
