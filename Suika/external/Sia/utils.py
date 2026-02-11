from Suika.utils import is_power_of_2

from Suika.external.Sia.hyperparam import *

import collections
from typing import List

# Parameters for a performance model which predicts the per-step time of
# distributed SGD using all-reduce. At a high level, models compute time and
# network time separately, and combines them with some degree of overlap.
# Compute time is modeled as a linear function of the local batch size.
# Network time is modeled using different parameters depending on if the job
# is inter-node (there exists a pair of replicas on different nodes), or
# intra-node (all replicas are on the same node). For both cases, network time
# is modeled as a constant term plus a retrogression term which increases
# linearly with the total number of replicas.
PerfParams = collections.namedtuple("PerfParams", [
    # T_compute ~ alpha_c + beta_c * local_bsz +
    #             (alpha_a + beta_a * local_bsz) * accumulation_steps
    "alpha_c",  # Constant term of compute time
    "beta_c",   # Multiplicative factor of compute time
    # If inter-node: T_network ~ alpha_n + beta_n * replicas
    "alpha_n",  # Constant term of inter-node network time
    "beta_n",   # Retrogression factor of inter-node network time
    # If intra-node: T_network ~ alpha_r + beta_r * replicas
    "alpha_r",  # Constant term of intra-node network time
    "beta_r",   # Retrogression factor of intra-node network time
    # T_step ~ (T_compute ^ gamma + T_network ^ gamma) ^ (1 / gamma)
    # Essentially is a p-norm where p = gamma. When p ~ 1 then
    # T_step ~ T_compute + T_network, indicating no overlap between compute
    # and network. When p -> infinity then T_step = max(T_compute, T_network),
    # indicating perfect overlap. We limit gamma to [1, 10] since 10 is close
    # enough to approximate the max function for our purposes.
    "gamma",    # Models the degree of overlap between compute and network
])

def _wrap_PerfParams(params):
    if isinstance(params, PerfParams):
        return params
    elif isinstance(params, (list, tuple)):
        return PerfParams(*params)
    elif isinstance(params, dict):
        return PerfParams(**params)
    else:
        raise TypeError(f"Unsupported type for params: {type(params)} - {params}. "
                        "Expected PerfParams, list, tuple, or dict.")

def is_valid_config(num_gpus, min_num_gpus=1) -> bool:
  assert num_gpus >= 0, "Number of GPUs must be non-negative."
  if num_gpus == 0:
    return True
  if num_gpus % min_num_gpus != 0:
    return False
  
  if num_gpus <= k_n_acc_per_node:
    return is_power_of_2(num_gpus)
  
  return num_gpus % k_n_acc_per_node == 0

def get_valid_configs(max_num_gpus = k_max_n_acc, min_num_gpus=1) -> List[int]:
  # 0 gpus is not included
  return [i for i in range(1, max_num_gpus + 1) 
            if is_valid_config(num_gpus=i, min_num_gpus=min_num_gpus)]

# logger
from Suika.utils import get_logger
import logging
sia_logger = get_logger("sia", console_level=logging.ERROR)
