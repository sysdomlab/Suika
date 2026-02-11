from typing import List, Tuple, Dict
import math

from Suika.cluster.estimator import Estimator
from Suika.cluster.plan_config import get_uniform_plan_config
from Suika.external.Rubick.utils import (
  PerfParams,
  EnvParams,
)
from Suika.external.Rubick.hyperparam import (
  k_acc_type,
  k_acc_TFLOPS,
  k_intra_bandwidth,
  k_inter_bandwidth,
  k_n_acc_per_node,
  get_rubick_rm,
  patch_rubick_tp_commu,
)
from Suika.external.Rubick.utils import rubick_logger
from Suika.models.utils import ModelConfig, ModelStrategy

import numpy as np
from scipy.optimize import minimize as scipy_minimize

class ThroughputFunction:
  
  atom_forward_time: float # time of the forward pass per sample per micro-batch
  total_param_size: float # total parameter size in MB
  model_config: ModelConfig # model config
  perf_params: PerfParams # performance model info
  estimator: Estimator # memory feasibility test only; latency is derived from Rubick's modeling
  
  skip_estimator: bool # skip feasibility test for Rubick
  
  """
  Args:
      model_config (ModelConfig): model config
      perf_params (PerfParams): performance model info
      skip_estimator (bool): skip estimator and feasibility test; only use it for fit_perf_params
      
    if skip_estimator is True,
      atom_forward_time and total_param_size are set to model_config values
    else:
      inferred from estimator_C
      
      (essentially, two cases should be the same.)
  
  Unit:
    time unit in ThroughputFunction is in seconds.
    but perf_params is left unchanged
  """
  def __init__(self, 
               model_config: ModelConfig,
               perf_params: PerfParams = None, 
               skip_estimator: bool = False, # skip estimator will not run the feasibility test, only use it for fit_perf_params
              ):
    # "comp" constant is multipled by TFLOPS in Suika, so this reverts it back
    self.atom_forward_time = model_config.estimator_C["comp"] / k_acc_TFLOPS \
      if not skip_estimator else model_config.atom_forward_time
    self.total_param_size = model_config.estimator_C["param"] * model_config.n_layer \
      if not skip_estimator else model_config.total_param_size
    
    assert self.atom_forward_time is not None, f"atom_forward_time is None"
    assert self.total_param_size is not None, f"total_param_size is None"
    
    self.model_config = model_config
    
    self.perf_params = perf_params if perf_params is not None else model_config.rubick_perf_params
    
    assert self.perf_params is not None, f"perf_params is None"
    
    self.estimator = Estimator(get_rubick_rm())
    if not skip_estimator:
      assert model_config.estimator_C is not None, "Estimator config is required for Rubick"
      self.estimator.set_constant_dict(model_config.estimator_C)
    
    self.skip_estimator = skip_estimator

    
  def __call__(self, 
               pp_dp_tp: Tuple[int, int, int], # pp dp tp
               global_batch_size: int, # batch size
               num_micro_batch: int, # number of microbatches
               output_latency: bool = False, # output latency instead of throughput
              ) -> float:
    """
    Predict Throughput
    
    Returns:
        float: per gpu throughput in form of global_batch_size / latency
        
        latency is in seconds
    """

    if not self.skip_estimator:
      # feasibility test
      plan_config = get_uniform_plan_config(
                acc_type=k_acc_type,
                pp_dp_tp=pp_dp_tp,
                gbs=global_batch_size,
                n_layer=self.model_config.n_layer,
                n_mb=num_micro_batch,
                max_tp_allowed=self.model_config.get_max_tp_allowed(),
              )
      if not self.estimator.feasible(plan_config):
        return 0.0 if not output_latency else float("inf")
      
      ## No longer needed, compact_allocation does that
      # if pp_dp_tp[1] * pp_dp_tp[2] <= k_n_acc_per_node:
      #   # set dp commu to intra node if applicable (for bandwidth consistency with rubick)
      #   for i, dp_mesh in enumerate(plan_config.acc_mesh):
      #     for tp_mesh in dp_mesh:
      #       for acc in tp_mesh:
      #         acc.node_id = i
          
      ref_latency = self.estimator.estimate(plan_config)
    
    pp, dp, tp = pp_dp_tp
    atom_batch_size = (global_batch_size + num_micro_batch * dp - 1)// (num_micro_batch * dp)
    
    # comp
    l = self.model_config.n_layer
    T_fwd = self.atom_forward_time * atom_batch_size * l / tp
    T_fwd = T_fwd / pp * (num_micro_batch + pp - 1)
    
    T_bwd = self.perf_params.k_bwd * T_fwd
    # GA/GC is not considered for 3D parallelism
    
    # commu
    V_dp = self.total_param_size * 2 * (dp - 1) / (dp * tp * pp)
    # total_param_size already in MB, no need to convert V_dp
    B_dp = k_intra_bandwidth if tp * dp <= k_n_acc_per_node else k_inter_bandwidth
    T_commu_dp = V_dp / B_dp
    
    b, s, h = global_batch_size, self.model_config.n_positions, self.model_config.n_embd
    if not patch_rubick_tp_commu:  
      V_tp = 4 * 2 * (tp - 1) * (b * s * h * l) / (dp * tp) # numel
    else:
      V_tp = 4 * 2 * (tp - 1) / tp * (atom_batch_size * s * h * l) / pp * (num_micro_batch + pp - 1) 

    V_tp = V_tp * 2 / 1024 / 1024 # FP16, to byte, then to MB
    B_tp = k_intra_bandwidth
    T_commu_tp = V_tp / B_tp
    
    V_pp = 2 * pp * (b * s * h) / (dp * tp) ** 2
    V_pp = V_pp * 2 / 1024 / 1024
    B_pp = k_inter_bandwidth
    T_commu_pp = V_pp / B_pp
    
    T_cc = T_fwd + self.overlap(T_bwd, T_commu_dp, self.perf_params.k_sync) + T_commu_tp + T_commu_pp
    
    k_opt = self.perf_params.k_opt / 1000 
    T_opt = k_opt * self.total_param_size / (tp * pp)
    # offload is not considered for 3D parallelism
    T_oo = T_opt
    
    k_const = self.perf_params.k_const / 1000
    T_iter = T_cc + T_oo + k_const
    
    # if not self.skip_estimator:
    #   # check for latency consistency 
    #   relative_error = abs(T_iter - ref_latency) / max(T_iter, ref_latency)
    #   if relative_error > 0.1:
    #     rubick_logger.warning(f"estimation mismatch greater than 10% for {pp_dp_tp}: Rubick {T_iter:.4f}; Suika {ref_latency:.4f}; relative error {relative_error*100:.1f}%")
    #   else:
    #     rubick_logger.debug(f"estimation match for {pp_dp_tp}: Rubick {T_iter:.4f}; Suika {ref_latency:.4f}; relative error {relative_error*100:.1f}%")
    
    if output_latency:
      return T_iter
    return global_batch_size / T_iter
  
  def predict_throughput_from_ModelStrategy(self, model_strategy: ModelStrategy, output_latency: bool = False) -> float:
    # assert model_strategy.model_config.rubick_perf_params == self.perf_params, f"PerfParams mismatch"
    
    return self.__call__(
      pp_dp_tp=model_strategy.pp_dp_tp,
      global_batch_size=model_strategy.global_batch_size,
      num_micro_batch=model_strategy.num_micro_batch,
      output_latency=output_latency,
    )
    
  @staticmethod
  def overlap(T1, T2, k):
    return math.pow(
      math.pow(T1, k) + math.pow(T2, k),
      1 / k
    )
    
def predict_throughput(
  model_strategy: ModelStrategy, # model strategy
  skip_estimator: bool = False, # skip estimator and feasibility test
  output_latency: bool = False, # output latency instead of throughput
) -> float:
  if model_strategy.acc_type != k_acc_type:
    raise ValueError(f"GPU_type mismatch {model_strategy.acc_type} {k_acc_type}")
  if model_strategy.model_config.rubick_perf_params is None:
    raise ValueError(f"PerfParams is None")
  if not skip_estimator and model_strategy.model_config.estimator_C is None:
    raise ValueError(f"Estimator config is None")
  if not model_strategy.validate():
    raise ValueError(f"Invalid model strategy {model_strategy}: {model_strategy.pp_dp_tp} {model_strategy.global_batch_size} {model_strategy.num_micro_batch}")
  
  th_fn = ThroughputFunction(
    model_config=model_strategy.model_config,
    # perf_params=model_strategy.model_config.rubick_perf_params,
    skip_estimator=skip_estimator,
  )
  throughput_or_latency = th_fn.predict_throughput_from_ModelStrategy(
    model_strategy=model_strategy,
    output_latency=output_latency,
  )
  return throughput_or_latency

"""
Fit performance parameters from profiled strategies and latencies

Args:
  profiled_strategies (List[ModelStrategy]): list of model strategies; perf_params field is ignored and fitted from scratch
  profiled_latencies (List[float]): list of latencies in seconds
  
Unit:
  passed in latencies are in seconds, but converted to ms during fitting, to comply with Rubick's original implementation
"""
def fit_perf_params(
  profiled_strategies: List[ModelStrategy],
  profiled_latencies: List[float], 
) -> PerfParams:
  profiled_latencies_ms = np.array([latency * 1000 for latency in profiled_latencies])
  
  # Order of parameters: k_bwd, k_sync, k_opt, k_os_offload, k_const, k_off, k_swap
  param_keys = ["k_bwd", "k_sync", "k_opt", "k_os_offload", "k_const", "k_off", "k_swap"]

  default_perf_params_dict = {
    "k_bwd": 1.5,
    "k_sync": 1.0,
    "k_opt": 1.0,
    "k_os_offload": 1.0, # not used in current 3D model
    "k_const": 1e-3,
    "k_off": 1.0, # not used in current 3D model
    "k_swap": 1.0, # not used in current 3D model
  }
  
  initial_params = [default_perf_params_dict[key] for key in param_keys]
  
  min_latency = min(profiled_latencies_ms)

  range_limits_dict = {
    "k_bwd": (1.0, 3.0),
    "k_sync": (1.0, 10.0),
    "k_opt": (1e-8, 100.0),
    "k_os_offload": (1.0, 1.0), # Fixed as not used
    "k_const": (1e-8, 100.0),
    "k_off": (1.0, 1.0),       # Fixed as not used
    "k_swap": (1.0, 1.0),      # Fixed as not used
  }
  bounds = [range_limits_dict[key] for key in param_keys]
  
  def set_perf_params_in_strategies(current_params_array):
    params_dict = {key: value for key, value in zip(param_keys, current_params_array)}
    current_perf_params = PerfParams(**params_dict)
    for strategy in profiled_strategies:
      # its fine if model_config is shared anyway.
      strategy.model_config.rubick_perf_params = current_perf_params
    
  def get_predicted_latencies_ms():
    predicted_latencies = []
    for strategy in profiled_strategies:
      # predict_throughput returns latency in seconds if output_latency is True
      latency_s = predict_throughput(
        model_strategy=strategy,
        skip_estimator=True,
        output_latency=True,
      )
      predicted_latencies.append(latency_s * 1000) # Convert to ms
    return np.array(predicted_latencies)

  def rmsle_loss(params_array):
    set_perf_params_in_strategies(params_array)
    predicted_latencies_ms = get_predicted_latencies_ms()
    
    log_predicted = np.log1p(predicted_latencies_ms)
    log_actual = np.log1p(profiled_latencies_ms)
    
    squared_errors = (log_predicted - log_actual) ** 2
    mean_squared_error = np.mean(squared_errors)
    rmsle = np.sqrt(mean_squared_error)
    
    if np.isnan(rmsle) or np.isinf(rmsle):
        return np.finfo(float).max # Return a large number if RMSLE is invalid
    return rmsle

  result = scipy_minimize(
    rmsle_loss,
    initial_params,
    method='L-BFGS-B',
    bounds=bounds,
    options={'ftol': 1e-7, 'gtol': 1e-6, 'maxiter': 500}
  )

  if not result.success:
    print(f"Warning: Optimization might not have converged. Message: {result.message}")

  optimized_params_dict = {key: value for key, value in zip(param_keys, result.x)}
  fitted_perf_params = PerfParams(**optimized_params_dict)
  
  return fitted_perf_params

