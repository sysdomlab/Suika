from Suika.cluster.job import Job
from Suika.cluster.estimator import AcceleratorMesh1D, AcceleratorMesh2D, Accelerator, AcceleratorMesh3D, sort_by_affinity

from Suika.cluster.plan_config import PlanConfig, get_uniform_plan_config

from typing import Callable, Optional, Tuple, List
import copy

from Suika.cluster.resource import ResourceManager
from Suika.utils import flatten_mesh

def merge_acc_mesh_by_DP(
  acc_mesh1: AcceleratorMesh3D,
  acc_mesh2: AcceleratorMesh3D,
) -> AcceleratorMesh3D:
  if acc_mesh1 is None or acc_mesh2 is None:
    return None
  
  pp_deg1 = len(acc_mesh1)
  dp_deg1 = len(acc_mesh1[0])
  tp_deg1 = len(acc_mesh1[0][0])
  
  pp_deg2 = len(acc_mesh2)
  dp_deg2 = len(acc_mesh2[0])
  tp_deg2 = len(acc_mesh2[0][0])
  
  assert pp_deg1 == pp_deg2 and tp_deg1 == tp_deg2, \
      f"Cannot merge acc_mesh with different pp_deg or tp_deg: {pp_deg1, dp_deg1, tp_deg1} vs {pp_deg2, dp_deg2, tp_deg2}"
  
  merged_acc_mesh = [
    acc_mesh1[i] + acc_mesh2[i]
    for i in range(pp_deg1)
  ]
  
  return merged_acc_mesh
  

def DP_optimize_all(
  job: Job,
  cur_plan_config: PlanConfig,
  new_acc_units: AcceleratorMesh2D,
  minimal_scaling: bool,
  rm: ResourceManager = None
) -> Tuple[List[Tuple[PlanConfig, float]], AcceleratorMesh2D]:
  self = job.job_optimizer
  
  if rm is None:
    rm = self.rm
  v_rm = copy.deepcopy(rm)
  
  # assume uniform partition
  acc_mesh = cur_plan_config.acc_mesh
  pp_deg = len(acc_mesh)
  dp_deg = len(acc_mesh[0])
  tp_deg = len(acc_mesh[0][0])

  cur_acc_mesh = cur_plan_config.acc_mesh
  new_acc_units = sort_by_affinity(cur_acc_mesh, new_acc_units, v_rm)

  avail_accs: List[Accelerator] = flatten_mesh(cur_acc_mesh)
  avail_new_accs: List[Accelerator] = []
  cur_world_size = len(avail_accs)
  
  f = [(None, -1)] * len(new_acc_units)
  
  for i, unit in enumerate(new_acc_units):
    avail_accs += flatten_mesh(unit)
    avail_new_accs += flatten_mesh(unit)
    
    new_world_size = len(avail_accs)
    if new_world_size % (pp_deg * tp_deg) != 0:
      continue
    if new_world_size > cur_world_size * 2:
      break
    
    new_dp_deg = new_world_size // pp_deg // tp_deg
    new_plan_config = get_uniform_plan_config(
      acc_type=avail_accs[0].acc_type_name(),
      pp_dp_tp=(pp_deg, new_dp_deg, tp_deg),
      n_layer=cur_plan_config.total_num_layer(),
      gbs=cur_plan_config.global_batch_size(),
      n_mb=cur_plan_config.num_micro_batch,
      fractional=False
    )
    
    ## unfortunately, this allocation may not respect the original topology... and is problematic when shrinking (which assume the original topology)
    ## we have to allocate the additional accelerators and merge it with the current acc_mesh as a workaround (don't want to refactor .allocate()...)
    # v_rm.raw_free(avail_accs, strict=False)
    # new_acc_mesh = v_rm.allocate(new_plan_config.acc_mesh, virtual_test=True, mask=avail_accs)
    
    additional_acc_mesh = get_uniform_plan_config(
      acc_type=avail_accs[0].acc_type_name(),
      pp_dp_tp=(pp_deg, new_dp_deg - dp_deg, tp_deg),
      gbs=1, n_layer=1, n_mb=1,
      fractional=True,
    ).acc_mesh
    
    additional_acc_mesh = v_rm.allocate(additional_acc_mesh, virtual_test=True, mask=avail_new_accs)
    new_acc_mesh = merge_acc_mesh_by_DP(cur_acc_mesh, additional_acc_mesh)
    
    if new_acc_mesh is not None:
      new_plan_config = new_plan_config.instantiate(new_acc_mesh)
      throughput = self.eval_throughput(new_plan_config)
      if throughput > 0:
        f[i] = (new_plan_config, throughput)
        
        if minimal_scaling: # minimum scaling, stop at the first feasible plan
          return f, new_acc_units

  return f, new_acc_units