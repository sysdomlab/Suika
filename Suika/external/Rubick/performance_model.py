from typing import Dict, List, Tuple
import copy

from Suika.external.Rubick.pred_throughput import ThroughputFunction
from Suika.external.Rubick.utils import PerfParams
from Suika.external.Rubick.hyperparam import (
  k_n_acc_per_node, 
  k_max_n_acc,
  perf_curve_min,
  perf_curve_max,
)

from Suika.models.utils import ModelConfig, ModelStrategy
from Suika.utils import is_power_of_2



class PerformanceModel():
  
  model_strategy: ModelStrategy # info of model_config, gbs, n_mb, and default plan
  
  th_fn: ThroughputFunction
  
  orig_th: float # original throughput of user provided plan
  
  perf_curve: Dict[int, Tuple[float, ModelStrategy]] # mapping num gpus to (max throughput, model_strategy), may not use all GPUs
  
  def __init__(self, model_strategy: ModelStrategy):
    
    self.model_strategy = model_strategy
        
    self.th_fn = ThroughputFunction(
      model_config=model_strategy.model_config,
      skip_estimator=False,
    )
    
    self.orig_th = self.th_fn.predict_throughput_from_ModelStrategy(
      model_strategy=model_strategy,
    )
    
    assert self.orig_th > 0, "original plan is not feasible"
    
    self.perf_curve = {
      i: (0.0, None)
      for i in range(perf_curve_min, perf_curve_max + 1)
    }
    
    self.gen_perf_curve()
    
  def predict_throughput_with_fixed_num_acc(self, num_acc: int) -> Tuple[float, ModelStrategy]:
    """
    Predict throughput with fixed number of GPUs
    with assumption that TP group is on the same node, while DP/PP group is no restricted.
    """
    if self.perf_curve[num_acc][0] > 0:
      # already calculated
      return self.perf_curve[num_acc]
    
    available_strategy = self.get_available_strategy(num_acc)
    for strategy in available_strategy:

      thp = self.th_fn.predict_throughput_from_ModelStrategy(
        model_strategy=strategy,
      )

      if thp > 0 and thp > self.perf_curve[num_acc][0]:
        self.perf_curve[num_acc] = (thp, strategy)
  
    return self.perf_curve[num_acc]
  
  def gen_perf_curve(self):
    """
    Generate performance curve for all GPUs
    """
    for num_acc in range(1, k_max_n_acc + 1):
      self.predict_throughput_with_fixed_num_acc(num_acc)
    
    # allow not using all gpus
    for num_acc in range(0, perf_curve_max + 1):
      if self.perf_curve[num_acc - 1][0] * (1 + 1e-2) > self.perf_curve[num_acc][0]:
        self.perf_curve[num_acc] = self.perf_curve[num_acc - 1]
    
    return self.perf_curve

  def get_available_strategy(self, num_acc: int) -> List[ModelStrategy]:
    available_strategy = []
    
    for pp in range(1, num_acc + 1):
      for dp in range(1, num_acc + 1):
        if not num_acc % (pp * dp) == 0:
          continue
        tp = num_acc // (pp * dp)
        if not is_power_of_2(tp) or tp > k_n_acc_per_node:
          continue
        
        model_strategy = copy.deepcopy(self.model_strategy)
        model_strategy.pp_dp_tp = (pp, dp, tp)
        # feasibility test is not conducted here. leave for ThroughputFunction
        if model_strategy.validate():
          available_strategy.append(model_strategy)
    
    return available_strategy