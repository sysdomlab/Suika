from .cpp_backend_impl import eval_throughput as eval_throughput_impl
from .cpp_backend_impl import optimize_all as optimize_all_impl

from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
  from Suika.cluster.estimator import AcceleratorMesh2D
  from Suika.cluster.resource import ResourceManager
  from Suika.cluster.plan_config import PlanConfig


"""
Evaluate Throughput with C++ backend. (primarily used for testing only)
"""
def eval_throughput(
  plan_config: "PlanConfig", 
  rm: "ResourceManager", 
  estimator_C: Dict[str, float],
  test_feasible: bool = True,
  # log_file: str = "log/eval_throughput_cpp.log",
  log_file: str = "",
) -> float:
  return eval_throughput_impl(plan_config, rm, estimator_C, test_feasible, log_file)

"""
Optimize All with C++ backend
"""

def optimize_all(
  cur_plan_config: "PlanConfig", 
  new_acc_units: "AcceleratorMesh2D",
  rm: "ResourceManager", 
  estimator_C: Dict[str, float],
  # log_file: str = "log/optimize_all_cpp.log",
  log_file: str = "",
) -> Tuple[List[Tuple["PlanConfig", float]], "AcceleratorMesh2D"]:
  pc_list, sort_id = optimize_all_impl(cur_plan_config, new_acc_units, rm, estimator_C, log_file)
  new_acc_units = [[]] + [
    new_acc_units[i] for i in sort_id[1:]
  ]
  return pc_list, new_acc_units