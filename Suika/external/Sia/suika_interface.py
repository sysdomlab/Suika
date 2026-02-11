from Suika.cluster.estimator import Estimator
from Suika.cluster.job import Job, JobStatus
from Suika.cluster.plan_config import PlanConfig, get_uniform_plan_config

from Suika.cluster.resource import get_default_rm
from Suika.external.Sia.goodput import GoodputFunction
from Suika.external.Sia.utils import (
  get_valid_configs,
  is_valid_config,
  sia_logger
)
from Suika.external.Sia.hyperparam import *
from Suika.external.Sia.milp import MILP_solve

import numpy as np
import copy
from typing import Callable, Dict, List, Optional, Union, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
  from Suika.cluster.scheduler import Scheduler

"""
  Get minimum number of GPUs required for a given PlanConfig. (pp_deg x tp_deg)
"""
def get_min_num_gpus(pc: PlanConfig) -> int:
  pp_deg = len(pc.acc_mesh)
  tp_deg = len(pc.acc_mesh[0][0])
  return pp_deg * tp_deg

"""
  PlanConfig to Sia plan descriptor
    (num_nodes, num_replicas, atomic_bsz, accum_steps)
  
  also a validity check tool for the (PlanConfig, num_gpus)
"""

def to_Sia_plan_descriptor(pc: PlanConfig, num_gpus:int =-1) -> Union[Tuple[int, int, int], None]:
  if pc is None or num_gpus == 0:
    return None
    
  if num_gpus == -1:
    num_gpus = pc.world_size()
    
  min_num_gpus = get_min_num_gpus(pc)
  pp_deg = len(pc.acc_mesh)
  gmbs = pc.global_micro_batch_size()
  
  if num_gpus % min_num_gpus != 0: # not divisible by min_num_gpus
    return None
  
  num_nodes = max(1, (num_gpus // pp_deg + k_n_acc_per_node - 1) // k_n_acc_per_node) # num_nodes is with respect to a single DP group, so div by pp_deg
  num_replicas = num_gpus // min_num_gpus # dp_deg
  atomic_bsz = (gmbs + num_replicas - 1) // num_replicas # ceil div
  accum_steps = pc.num_micro_batch - 1
  
  if gmbs // num_replicas < 1: # num_replicas too large
    return None

  return (num_nodes, num_replicas, atomic_bsz, accum_steps)
  
"""
  Eval PlanConfig throughput with goodput_fn
  
  Guarded with Suika's feasibility check
"""
def evaluate_goodput_from_plan_config(pc: PlanConfig, # reference pc, dp_degree will be determined by num_gpus
                                      goodput_fn: GoodputFunction, 
                                      num_gpus: int = -1,
                                      estimator_C: Dict[str, float] = None
                                      ) -> float:

  plan_descriptor = to_Sia_plan_descriptor(pc, num_gpus)
  if plan_descriptor is None:
    return 0.0
  num_nodes, num_replicas, atomic_bsz, accum_steps = plan_descriptor
  
  if estimator_C:
    estimator = Estimator(get_default_rm(), estimator_C)
    actual_pc = convert_Sia_plan_to_Suika_plan(pc, num_gpus)
    if not estimator.feasible(actual_pc):
      return 0.0

  return goodput_fn.evaluate(
    num_nodes=num_nodes,
    num_replicas=num_replicas,
    atomic_bsz=atomic_bsz,
    accum_steps=accum_steps
  )


"""
  Convert a Sia plan to a Suika plan.
  
  Sia plan essentially encompasses:
    1. atomic pipeline info, including pp_deg, tp_deg, n_mb, n_layer, and global bsz (fixed)
    2. dp_deg (free)
    
  We wrap 1. with PlanConfig, and 2. with num_gpus / tp_deg / pp_deg.
    (use num_gpus instead of dp_deg for MILP formulation consistency)
"""
def convert_Sia_plan_to_Suika_plan(pc: PlanConfig, num_gpus: int) -> Optional[PlanConfig]:
  if pc is None or num_gpus == 0: 
    # by the way, one should refer to Job.usr_plan_config instead of cur_plan_config which may be None.
    return None
  
  assert is_valid_config(num_gpus) and to_Sia_plan_descriptor(pc, num_gpus) is not None, \
    f"Attempt to convert plan {pc} with {num_gpus} gpus that is not valid"
  
  pp_deg = len(pc.acc_mesh)
  tp_deg = len(pc.acc_mesh[0][0])
  dp_deg = num_gpus // (pp_deg * tp_deg)

  pp_dp_tp = (pp_deg, dp_deg, tp_deg)
  
  # adjust node_id to comply with Sia's placement policy: 
  # - either take a partial node by 2^k accs, or take whole node(s). (this is always possible, guaranteed by Sia's valid config constraint)
  # - node_id takes effect when ResourceManager.allocate is called
  #   (acc with same node_id in virtual plan are forced to be placed on the same physcial node)
  # - given that tp_deg <= 8 and is power of 2 => tp_deg | 8. so, this rearrangement won't leave same tp group on different nodes.
  # - compact_allocation argument forces this constraint.
  new_pc = get_uniform_plan_config(acc_type=k_acc_type, 
                                  pp_dp_tp=pp_dp_tp, 
                                  gbs=pc.global_batch_size(),
                                  n_layer=pc.total_num_layer(),
                                  n_mb=pc.num_micro_batch,
                                  is_micro_gbs=False,
                                  fractional=False,
                                  compact_allocation=True,
                                  n_acc_per_node=k_n_acc_per_node
                                  )
  return new_pc

class PolicySiaImpl:
  
  sch: "Scheduler"
  cached_decisions: List[Union[Tuple[Job, PlanConfig], None]]
  
  def __init__(self, 
               sch: "Scheduler", 
               p_fairness: float = k_p_fairness,
               lambda_a: float = k_lambda_a,
               lambda_n: float = k_lambda_n,
              ):
    self.sch = sch
    self.cached_decisions = []
    
    self.p_fairness = p_fairness
    self.lambda_a = lambda_a
    self.lambda_n = lambda_n
    
  def submit(self, job: Job):
    # good_fn is set at get_goodput_matrix
    pass
  
  def schedule(self) -> Union[Tuple[Job, PlanConfig], None]:
    
    if not self.cached_decisions:
      active_jobs = self.sch._get_active_jobs()
      virtual_decisions = \
        self.optimize(active_jobs=active_jobs, 
                      total_num_gpus=self.sch.rm.get_total_acc_num(),
                      p_fairness = self.p_fairness,
                      lambda_a=self.lambda_a,
                      lambda_n=self.lambda_n,
                      get_job_age=self.sch._get_job_age,
                      get_job_avg_reconfig_cost=self.sch._get_job_avg_reconfig_cost,
                      get_job_reconfig_cnt=self.sch._get_job_reconfig_cnt,
                    )
      self.cached_decisions = self._virtual_decisions_to_phyiscal(virtual_decisions)
        
        
      
    while self.cached_decisions:
      job, plan_config = self.cached_decisions.pop(0)
      if job.status in [JobStatus.RUNNING, JobStatus.WAITING]:
        action = ("launch" if job.status == JobStatus.WAITING else "reconfig") if plan_config else "preempt"
        sia_logger.info(f"schedule(): {job}@{action}")
        return (job, plan_config)
      else:
        sia_logger.warning(f"schedule() failed: {job} is not in RUNNING or WAITING state; cached decision discarded")
        self._discard_cache()
        return None
      
    return None
  
  @staticmethod
  def optimize(
              active_jobs: List[Job], 
              total_num_gpus: int, 
              p_fairness: float,
              lambda_a: float,
              lambda_n: float,
              get_job_age: Callable[[Job], float], # pass in as custom function, for better testability
              get_job_avg_reconfig_cost: Callable[[Job], float], 
              get_job_reconfig_cnt: Callable[[Job], int],
              ) -> List[Tuple[Job, PlanConfig]]:
    sia_logger.debug("optimize(): start optimization with " \
                    f"{len(active_jobs)} active jobs, " \
                    f"total_num_gpus={total_num_gpus}, " \
                    f"p_fairness={p_fairness}, lambda_a={lambda_a}, lambda_n={lambda_n}")
    sia_logger.debug(f"optimize(): jobs {active_jobs}")
    
    if not len(active_jobs):
      return []
    
    # step 1: populate goodput matrix (goodput_fn generated inside)
    goodput_matrix, prev_allocation, idx2job, idx2cf, job2idx, cf2idx = PolicySiaImpl._get_goodput_matrix(
      active_jobs, 
      total_num_gpus=total_num_gpus,
      p_fairness=p_fairness,
      get_job_age=get_job_age,
      get_job_avg_reconfig_cost=get_job_avg_reconfig_cost,
      get_job_reconfig_cnt=get_job_reconfig_cnt
      )  # dim (n_jobs, n_configs)

    # step 2: call MILP optimize
    # allocation is a (n_jobs, n_configs) binary matrix, where allocation[i, j] = 1 means job i is allocated to config j.
    allocation = MILP_solve(goodput_matrix=goodput_matrix,
                            prev_allocation=prev_allocation,
                            total_num_gpus=total_num_gpus,
                            p_fairness=p_fairness,
                            lambda_a=lambda_a,
                            lambda_n=lambda_n
                          )
        
    # step 3: convert MILP solutions (num_gpus for each job) to PlanConfig, and scheduling decisions
    virtual_decisions = []
    for i0, job in enumerate(idx2job.values()):
      job: Job
      if (allocation[i0] == prev_allocation[i0]).all():
        continue
      
      if allocation[i0].sum() == 0:
        # preempted
        pc = None
        num_gpus = 0
      else: 
        # allocate to a new plan
        i1 = allocation[i0].argmax()
        num_gpus = idx2cf[i1]
        pc = convert_Sia_plan_to_Suika_plan(job.usr_plan_config, num_gpus)
      
      sia_logger.info(f"optimize(): {job} allocated {num_gpus} gpus")
      virtual_decisions.append((job, pc))
    
    return virtual_decisions
  
  @staticmethod
  def _get_goodput_matrix(jobs: List[Job],
                          total_num_gpus: int,
                          p_fairness: float,
                          get_job_age: Callable[[Job], float], # pass in as custom function, for unit test without actual scheduler
                          get_job_avg_reconfig_cost: Callable[[Job], float], 
                          get_job_reconfig_cnt: Callable[[Job], int],
                          ):
    
    # generate GoodputFunction for each job if not present
    #  set goodput_fn and estimator_C in job.other_context
    for job in jobs:
      if "goodput_fn" not in job.other_context:
        sia_perf_params = None
        estimator_C = None
        
        if "model_strategy" in job.other_context:
          model_config = job.other_context["model_strategy"].model_config
          sia_perf_params = model_config.sia_perf_params
          estimator_C = model_config.estimator_C
          
        assert sia_perf_params is not None, \
          f"Missing sia_perf_params for {job}"
        
        goodput_fn = GoodputFunction(sia_perf_params)
        job.other_context["goodput_fn"] = goodput_fn
        job.other_context["estimator_C"] = estimator_C
    
    # get index mapping
    idx2job = {i: job for i, job in enumerate(jobs)} # dim 0 of goodput matrix
    job2idx = {job: i for i, job in idx2job.items()}
      
    valid_configs = get_valid_configs(total_num_gpus)
    idx2cf = {i: cf for i, cf in enumerate(valid_configs)}  # dim 1 of goodput matrix
    cf2idx = {cf: i for i, cf in idx2cf.items()}
    
    # raw goodput matrix
    goodput_matrix = np.zeros((len(jobs), len(valid_configs)), dtype=np.float32)
    prev_allocation = np.zeros_like(goodput_matrix)
    num_min_gpus_matrix = np.zeros_like(goodput_matrix)
    
    for i0, job in idx2job.items():
      job: Job
      ref_pc = job.usr_plan_config
      goodput_fn: GoodputFunction = job.other_context["goodput_fn"]
      estimator_C: Dict[str, float] = job.other_context["estimator_C"]
      
      num_min_gpus = get_min_num_gpus(job.usr_plan_config)
      num_min_gpus_matrix[i0, :] = num_min_gpus
      prev_num_gpus = job.cur_plan_config.world_size() if job.cur_plan_config is not None else 0
            
      for i1, num_gpus in idx2cf.items():
        goodput = evaluate_goodput_from_plan_config(pc=ref_pc, 
                                                    goodput_fn=goodput_fn, 
                                                    num_gpus=num_gpus, 
                                                    estimator_C=estimator_C)
        
        # goodput may be zero
        # zero is the placeholder for invalid config in the code logic
          
        goodput_matrix[i0, i1] = goodput
          
        if prev_num_gpus == num_gpus:
          prev_allocation[i0, i1] = 1.0
    
    if DETAILED_LOGGING:
      sia_logger.debug(f"_get_goodput_matrix(): raw goodput_matrix=\n{goodput_matrix}")
      sia_logger.debug(f"_get_goodput_matrix(): prev_allocation=\n{prev_allocation}")
    
    # normalize
    rowwise_min_goodput_matrix = np.min(
      np.where(goodput_matrix > 0.0, goodput_matrix, INF),
      axis=-1,
      keepdims=True
    ) # minimum above 0
    goodput_matrix = num_min_gpus_matrix * goodput_matrix / rowwise_min_goodput_matrix
    
    if DETAILED_LOGGING:
      sia_logger.debug(f"_get_goodput_matrix(): normalized goodput_matrix=\n{goodput_matrix}")
    
    # restart factor
    """
      Sia uses a constant restart time for each job (which is indeeded reasonable) to compute r as (AGE - N * S) / (AGE + S).
      where, N = number of restarts, S = restart time  
      Our scheduler have interface to retrieve AGE, N, S directly, so we use that instead of pre-profiling S.
    """
    
    restart_factor_matrix = np.ones_like(goodput_matrix, dtype=np.float32)
    
    for i0, job in idx2job.items():
      age = get_job_age(job) + 1e-3 # avoid divided by zero
      avg_reconfig_cost = get_job_avg_reconfig_cost(job)
      reconfig_cnt = get_job_reconfig_cnt(job)
            
      restart_factor_matrix[i0, :] = (age - reconfig_cnt * avg_reconfig_cost) / (age + avg_reconfig_cost)
      
      if job.cur_plan_config is not None:
        cur_num_gpus = job.cur_plan_config.world_size()
        assert cur_num_gpus in valid_configs, f"Current number of gpus {cur_num_gpus} is not in valid configs {valid_configs}"
        i1 =  cf2idx[cur_num_gpus]
        restart_factor_matrix[i0, i1] = 1.0
    
    restart_factor_matrix = np.maximum(0.0, restart_factor_matrix)  # ensure no negative values
    
    goodput_matrix = goodput_matrix * restart_factor_matrix
    
    if DETAILED_LOGGING:
      sia_logger.debug(f"_get_goodput_matrix(): restart_factor_matrix=\n{restart_factor_matrix}")
      sia_logger.debug(f"_get_goodput_matrix(): restart_factor discounted goodput_matrix =\n{goodput_matrix}")
    
    # fairness nob
    exponent = np.where(goodput_matrix == 0.0, 1.0, p_fairness) # avoid raising 0 to a negative exponent
    goodput_matrix = np.power(goodput_matrix, exponent)
    goodput_matrix = np.where( # set infeasible plans to -INF or INF
      goodput_matrix == 0.0, -INF if p_fairness > 0 else INF,
      goodput_matrix
    )
    if DETAILED_LOGGING:
      sia_logger.debug(f"_get_goodput_matrix(): final goodput_matrix=\n{goodput_matrix}")
    
    return goodput_matrix, prev_allocation, \
           idx2job, idx2cf, job2idx, cf2idx


  def _virtual_decisions_to_phyiscal(self, virtual_decisions: List[Tuple[Job, PlanConfig]]) -> List[Tuple[Job, PlanConfig]]:
    """
     Apply placement policy to convert virtual decisions to physical decisions.
    """
    shrink = []
    expand = []
    v_rm = copy.deepcopy(self.sch.rm)
    
    def get_ws(pc: PlanConfig) -> int:
      if pc is None:
        return 0
      return pc.world_size()
    
    for job, pc in virtual_decisions:
      if get_ws(pc) < get_ws(job.cur_plan_config):
        shrink.append((job, pc))
      elif get_ws(pc) > get_ws(job.cur_plan_config):
        expand.append((job, pc))
      else:
        sia_logger.warning(f"_virtual_decisions_to_phyiscal(): job {job} is not reconfigured btw {job.cur_plan_config} and {pc}, but still in virtual decisions")
      
    phyiscal_decisions = []
    for job, pc in shrink:
      cur_acc_mesh = job.cur_plan_config.acc_mesh
      v_rm.free(cur_acc_mesh)
      
      if pc is not None:
        new_acc_mesh = v_rm.allocate(pc.acc_mesh, mask=cur_acc_mesh)
        assert new_acc_mesh is not None, f"Failed to allocate {job} at shrinking, pc = {pc}, usage = {v_rm.get_node_usage_dict()}" 
        pc = pc.instantiate(new_acc_mesh)
        
      phyiscal_decisions.append((job, pc))
    
    for job, pc in expand:
      
      if job.cur_plan_config is not None:
        cur_acc_mesh = job.cur_plan_config.acc_mesh
        v_rm.free(cur_acc_mesh)
      else:
        cur_acc_mesh = []
        
      acc_mesh = v_rm.allocate(pc.acc_mesh)
      if acc_mesh is None:
        v_rm.raw_allocate(cur_acc_mesh)
        continue
      pc = pc.instantiate(acc_mesh)
      
      phyiscal_decisions.append((job, pc))
    
    sia_logger.debug("-BEGINNING of PHYSICAL DECISIONS-------------------------------------------")
    for dec in phyiscal_decisions:
      sia_logger.debug(f"{dec}")
    sia_logger.debug("-END of PHYSICAL DECISIONS-------------------------------------------------")
    return phyiscal_decisions
  
  def _discard_cache(self):
    self.cached_decisions = []