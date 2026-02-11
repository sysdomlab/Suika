from collections import defaultdict
from typing import Dict, List, Tuple, Union, Callable, TYPE_CHECKING
import logging
import copy
import math

from Suika.cluster.resource import ResourceManager

if TYPE_CHECKING:
  from Suika.cluster.scheduler import Scheduler
  from Suika.external.Rubick.suika_interface import PolicyRubickImpl
  from Suika.external.Sia.suika_interface import PolicySiaImpl

from Suika.cluster.estimator import AcceleratorMesh1D, AcceleratorMesh2D, partition_acc, sort_by_affinity
from Suika.cluster.job import Job, JobStatus
from Suika.cluster.plan_config import PlanConfig, convert_to_rank_mesh_respectfully

import os

"""
Suika Scheduling Policy Hyperparameters

1. SUIKA_POLICY_TH_EXP: float
  the exponent to raise pessmisitc utilization to as the threshold for reconfiguration
  minimum effective speed-up threshold for reconfiguration
  lower the value, better the fairness
  
2. SUIKA_POLICY_HARD_SCALE_LIMIT: int
  the maximum ratio a job can scale up to compared to it's original plan 

3. SUIKA_POLICY_PREEMPT: bool
  whether the scheduling is preemptive (enabling tentative_preempt functionality)

4. SUIKA_POLICY_MULTITHREAD: bool
  whether the scheduling algorithm is multithreaded. (default to True; no impact on scheduling decision, just speed up)
  
5. SUIKA_POLICY_MULTIROUND: bool
  whether to enable multiround reconfiguration caching. (default to True)

6. SUIKA_POLICY_RECONFIG_PREEMPT: bool
  whether running can preempt other jobs. (default to True)
  > waiting jobs can always preempt if SUIKA_POLICY_PREEMPT is True
  > running jobs can only preempt if both SUIKA_POLICY_PREEMPT and SUIKA_POLICY_RECONFIG_PREEMPT are True

"""

SUIKA_POLICY_ENV_VAR_LIST = [
  "SUIKA_POLICY_TH_EXP",
  "SUIKA_POLICY_HARD_SCALE_LIMIT",
  "SUIKA_POLICY_PREEMPT",
  "SUIKA_POLICY_MULTITHREAD",
  "SUIKA_POLICY_MULTIROUND",
  "SUIKA_POLICY_RECONFIG_PREEMPT",
]

SUIKA_POLICY_TH_EXP = float(os.getenv("SUIKA_POLICY_TH_EXP", "1"))
SUIKA_POLICY_HARD_SCALE_LIMIT = int(os.getenv("SUIKA_POLICY_HARD_SCALE_LIMIT", "8"))
SUIKA_POLICY_PREEMPT = os.getenv("SUIKA_POLICY_PREEMPT", "true").lower() in ("true", "1", "yes", "y")
SUIKA_POLICY_MULTITHREAD = os.getenv("SUIKA_POLICY_MULTITHREAD", "true").lower() in ("true", "1", "yes", "y")
SUIKA_POLICY_MULTIROUND = os.getenv("SUIKA_POLICY_MULTIROUND", "true").lower() in ("true", "1", "yes", "y")
SUIKA_POLICY_RECONFIG_PREEMPT = os.getenv("SUIKA_POLICY_RECONFIG_PREEMPT", "true").lower() in ("true", "1", "yes", "y")

# if any(os.getenv(env_var, None) is not None for env_var in SUIKA_POLICY_ENV_VAR_LIST):
#   print("Suika Policy Hyperparameters:")
#   print(f"  SUIKA_POLICY_TH_EXP: {SUIKA_POLICY_TH_EXP}")
#   print(f"  SUIKA_POLICY_HARD_SCALE_LIMIT: {SUIKA_POLICY_HARD_SCALE_LIMIT}")
#   print(f"  SUIKA_POLICY_PREEMPT: {SUIKA_POLICY_PREEMPT}")
#   print(f"  SUIKA_POLICY_MULTITHREAD: {SUIKA_POLICY_MULTITHREAD}")
#   print(f"  SUIKA_POLICY_MULTIROUND: {SUIKA_POLICY_MULTIROUND}")
#   print(f"  SUIKA_POLICY_RECONFIG_PREEMPT: {SUIKA_POLICY_RECONFIG_PREEMPT}")

SUIKA_SIMULATION = os.getenv("SUIKA_SIMULATION", "False").lower() in ["true", "yes", "y", "1"]
# if os.getenv("SUIKA_SIMULATION", None) is not None:
#   print(f"SUIKA_SIMULATION: {SUIKA_SIMULATION}")

# set policy hyperparameters
def set_policy_threshold_exp(th_exp: int):
  global SUIKA_POLICY_TH_EXP
  SUIKA_POLICY_TH_EXP = th_exp
def set_policy_hard_scale_limit(hard_scale_limit: int):
  global SUIKA_POLICY_HARD_SCALE_LIMIT
  SUIKA_POLICY_HARD_SCALE_LIMIT = hard_scale_limit
def set_policy_preemption(flag: bool):
  global SUIKA_POLICY_PREEMPT
  SUIKA_POLICY_PREEMPT = flag
def set_suika_policy_multithread(flag: bool):
  global SUIKA_POLICY_MULTITHREAD
  SUIKA_POLICY_MULTITHREAD = flag
def set_suika_policy_multiround(flag: bool):
  global SUIKA_POLICY_MULTIROUND
  SUIKA_POLICY_MULTIROUND = flag
def set_suika_policy_reconfig_preempt(flag: bool):
  global SUIKA_POLICY_RECONFIG_PREEMPT
  SUIKA_POLICY_RECONFIG_PREEMPT = flag

"""
Policy Implementation: the real scheduling policy.
"""

"""
Abstract Policy Class
"""

class Policy():
  sch: "Scheduler"
  
  def __init__(self, scheduler: 'Scheduler'):
    self.sch = scheduler
  
  # add a new job
  def submit(self, job: Job):
    raise NotImplementedError("submit() not implemented")
  
  # get next schedule decision
  def schedule(self) -> Union[Tuple[Job, PlanConfig], None]:
    raise NotImplementedError("schedule() not implemented")
  
  def launch_success_callback(self, job: Job, plan_config: PlanConfig):
    pass
  
  def launch_failure_callback(self, job: Job, plan_config: PlanConfig):
    pass


"""
Dummy Policy Class
Never schedule anything for debug use.
"""
class PolicyDummy(Policy):
  
  def __init__(self, scheduler: 'Scheduler'):
    super().__init__(scheduler)

  def submit(self, job: Job):
    return
  
  def schedule(self):
    return None

"""
  Base Policies (handling waiting jobs)
"""

class PolicyBase(Policy):
  allow_reconfig: bool

  def __init__(self, scheduler: 'Scheduler', allow_reconfig):
    super().__init__(scheduler=scheduler)
    self.allow_reconfig = allow_reconfig

  def schedule(self, rm: ResourceManager = None) -> Union[Tuple[Job, PlanConfig], None]:
    if rm is None: # allow default rm=self.sch.rm is only for interface compatibility, PolicySuika never call this function with rm=None
      rm = self.sch.rm
    decision = self._schedule(rm=rm)
    if decision is not None:
      return decision

    if not self.allow_reconfig:
      return None
    else:
      return self._reconfig(rm=rm)
  
  def _schedule(self, rm: ResourceManager) -> Union[None, Tuple[Job, PlanConfig]]:
    # base scheduling policy without reconfig
    # if rm provided, assume to be virtual. internal state of policy should not be changed
    # otherwise, assume to be physical, modify internal state
    raise NotImplementedError("_schedule not implemented")

  # reconfig
  # should be implemented in the child class
  def _reconfig(self, rm: ResourceManager) -> Union[Tuple[Job, PlanConfig], None]:
    raise NotImplementedError("reconfig() not implemented")

"""
  First-Come, First-Served (FCFS) Policy
  with support for reconfiguration.
"""
class PolicyFCFS(PolicyBase):
  
  def __init__(self, scheduler: 'Scheduler', allow_reconfig: bool = False):
    super().__init__(scheduler=scheduler, allow_reconfig=allow_reconfig)
    
  def submit(self, job: Job):
    job.other_context["submit_time"] = self.sch._record_time()
    self.queue.append(job)

  def _schedule(self, rm: ResourceManager) -> Union[None, Tuple[Job, PlanConfig]]:

    for idx, job in enumerate(self.queue):
      job: Job
      acc_mesh = rm.allocate(job.usr_plan_config.acc_mesh, virtual_test=True)
      if acc_mesh is not None:
        plan_config = job.usr_plan_config.instantiate(acc_mesh)
        return (job, plan_config)
    
    return None
  
  @property
  def queue(self):
    def sort_key(job: Job):
      return job.other_context['submit_time']
    return list(sorted(self.sch._get_waiting_jobs(), key=lambda x: sort_key(x), reverse=False))
  
"""
  Shortest Job First (SJF) Policy
  with support for reconfiguration.
"""
class PolicySJF(PolicyBase):
  
  def __init__(self, scheduler: 'Scheduler', allow_reconfig: bool = False):
    super().__init__(scheduler=scheduler, allow_reconfig=allow_reconfig)
    
  def submit(self, job: Job):
    assert 'n_iter' in job.other_context, f"Job {job} must have 'n_iter' in other_context for SJF"
    assert 'iter_time' in job.other_context, f"Job {job} must have 'iter_time' in other_context for SJF"
    pass
    
  def _schedule(self, rm: ResourceManager) -> Union[None, Tuple[Job, PlanConfig]]:
    
    for idx, job in enumerate(self.queue):
      job: Job
      acc_mesh = rm.allocate(job.usr_plan_config.acc_mesh, virtual_test=True)
      if acc_mesh is not None:
        plan_config = job.usr_plan_config.instantiate(acc_mesh)
        return (job, plan_config)
    
    return None

  @property
  def queue(self):
    def sort_key(job: Job):
      return job.other_context['n_iter'] * job.other_context['iter_time']
    return list(sorted(self.sch._get_waiting_jobs(), key=lambda x: sort_key(x), reverse=False))

policy_base_impl = os.getenv("SUIKA_POLICY_BASE", "FCFS").lower()
if policy_base_impl in ["fcfs", "fifo"]:
  # print("Using base policy FCFS")
  PolicyBaseImpl = PolicyFCFS
elif policy_base_impl in ["sjf"]:
  print("Using base policy SJF")
  PolicyBaseImpl = PolicySJF
else:
  assert False, f"unkown policy base {policy_base_impl}"
    
"""
  Reconfigurable Policies
"""    

"""
  Suika Policy (ours)
  base policy can be changed by setting the base class to PolicyFCFS or PolicySJF.
"""
class PolicySuika(PolicyBaseImpl):
  
  cached_decisions: List[Tuple[Job, PlanConfig]] # for preemptions
  cached_chain_decisions: Dict[Job, List[PlanConfig]] # for multi-round scheduling
  prev_configs: Dict[Job, List[PlanConfig]]
  
  tentative_preempt_trial_interval: int
  tentative_preempt_trial_last: int

  def __init__(self, scheduler: 'Scheduler'):
    super().__init__(scheduler=scheduler, allow_reconfig=True)
    self.sch.log("PolicySuika initialized")
    self.prev_configs = {}
    self.cached_decisions = []
    self.cached_chain_decisions = {}
    self.tentative_preempt_trial_interval = self.sch.schedule_interval / 2 if not SUIKA_POLICY_MULTIROUND else 0 # in multi round reconfig, the tentative_preempt is only performed once per scheduling round, so no need to add a interval.
    self.tentative_preempt_trial_last = 0
  
  def schedule(self):
    # cache decision found
    if self.cached_decisions:
      return self.cached_decisions.pop(0)
    
    # try scheduling waiting job first
    decision = self._schedule(rm = self.sch.rm)
    if decision is not None:
      return decision

    # if no valid waiting job found, try tentative preempt
    now = self.sch._record_time()
    if SUIKA_POLICY_PREEMPT and \
      now - self.tentative_preempt_trial_last > self.tentative_preempt_trial_interval:
      if self._tentative_preempt():
        return self.schedule()
      else:
        self.tentative_preempt_trial_last = now
    
    # if no waiting job can be scheduled even with preemption
    # multi round reconfig
    # remark: self._reconfig relies on self.sch._get_running_jobs, so it's not safe to interleave launch of waiting jobs and reconfiguration of running jobs
    self._discard_cache()
    v_rm = copy.deepcopy(self.sch.rm)
    
    # cache cur_plan_config
    job_to_plan_config = {
      job: job.cur_plan_config for job in self.sch._get_active_jobs()
    }

    while True:
      decision = self._reconfig(rm=v_rm, job_to_plan_config=job_to_plan_config)
      if decision is None:
        break
      
      job, plan_config = decision
      cur_plan_config = job_to_plan_config[job]
      cur_acc_mesh = getattr(cur_plan_config, 'acc_mesh', [])
      new_acc_mesh = getattr(plan_config, 'acc_mesh', [])
      assert v_rm.raw_free(cur_acc_mesh, strict=True), \
        f"Suika policy schedule() failed to raw free {job} {cur_acc_mesh} while scheduling {plan_config}"
      assert v_rm.raw_allocate(new_acc_mesh, strict=True), \
        f"Suika policy schedule() failed to raw allocate {job} {new_acc_mesh} while scheduling {plan_config}"

      assert len(cur_acc_mesh) > 0 and cur_plan_config.world_size() <= plan_config.world_size(), f"Suika policy schedule(): not scale up between {cur_plan_config} to {plan_config}"

      job_to_plan_config[job] = plan_config

      if job not in self.cached_chain_decisions:
        self.cached_chain_decisions[job] = []
      self.cached_chain_decisions[job].append(plan_config)
      
      if not SUIKA_POLICY_MULTIROUND:
        break

    self.cached_decisions = list(map(tuple, [
      (job, pcs[-1]) for job, pcs in self.cached_chain_decisions.items()
    ])) # all decisions here are scale up, so order does not matter
    
    if len(self.cached_decisions):
      return self.schedule()
    else:
      return None

  def launch_success_callback(self, job: Job, plan_config: PlanConfig):
    
    # case 1. multi round reconfig compressed into one, unfold the chain
    # case 2. not multi round reconfig, just use given plan_config
    for plan_config in self.cached_chain_decisions.get(job, [plan_config]):
      if job not in self.prev_configs:
        # launch
        self.prev_configs[job] = []
        self.prev_configs[job].append(plan_config)
      elif self.prev_configs[job][-1].world_size() < plan_config.world_size():
        # scale up
        self.prev_configs[job].append(plan_config)
      else:
        # scale down or no scale (parallelism change)
        while self.prev_configs[job][-1].world_size() > plan_config.world_size():
          self.prev_configs[job].pop(-1)
    
    if job in self.cached_chain_decisions:
      del self.cached_chain_decisions[job]
          
  def launch_failure_callback(self, job: Job, plan_config: PlanConfig):
    self._discard_cache()
  
  def _reconfig(self, 
               rm: ResourceManager,
               optimize_all_fn_by_job:Callable[[Job], Callable]=None, # Job -> optimize_all_fn (for baseline DP/Metis plug-in)
               threshold: float = None,
               job_to_plan_config: Dict[Job, PlanConfig] = None,
              ) -> Union[None, Tuple[Job, PlanConfig]]:
    # prepare default parameters
    if optimize_all_fn_by_job is None:
      optimize_all_fn_by_job = lambda job: None

    running_jobs: List[Job] = self.sch._get_running_jobs(locking=False)

    if job_to_plan_config is None:
      job_to_plan_config = {}
    else:
      job_to_plan_config = copy.copy(job_to_plan_config)
    for job in running_jobs:
      if job not in job_to_plan_config:
        job_to_plan_config[job] = job.cur_plan_config
        
    if threshold is None:
      threshold = self._get_threshold(rm)
      
    # eval effective speedup
    acc_list = rm.get_avail_acc_list()
    if len(acc_list) == 0:
      return None

    best = None # (best job, plan_config)
    best_effective_speed_up = 0

    eval_results: Dict[Job, Tuple[Union[PlanConfig, None], float]] = {}
    
    def eval_worker(job: Job, acc_list: AcceleratorMesh1D):
      nonlocal eval_results
      if job.status != JobStatus.RUNNING:
        eval_results[job] = (None, -1.0)
      
      # we provides all jobs with complete acc_units,
      # but the job may choose to use only part of it, or rearrange it.
      if not SUIKA_SIMULATION: 
        # physical cluster, 64 GPUs, no need to speed up by clustering
        acc_units = partition_acc(acc_list, 1)
      else:
        # simulation, may have more 
        search_scale = min(len(acc_list), job.usr_plan_config.world_size())
        acc_units_granularity = 1
        if search_scale < 32:
          acc_units_granularity = 1          
        elif search_scale < 64:
          acc_units_granularity = 2
        elif search_scale < 128:
          acc_units_granularity = 4
        else:
          acc_units_granularity = 8
        acc_units = partition_acc(acc_list, acc_units_granularity)
        # enable limited expansion
        if len(acc_units):
          acc_units = sort_by_affinity(cur_acc_mesh=job_to_plan_config[job].acc_mesh, 
                                       new_acc_units=acc_units, 
                                       rm=rm)
          truncate_index = round(SUIKA_POLICY_HARD_SCALE_LIMIT * job.cur_plan_config.world_size() / acc_units_granularity + 0.5)
          acc_units = acc_units[:truncate_index + 1]
        
      new_plan_config, effective_speed_up = \
        job.eval_best_effective_speed_up(new_resources=acc_units,
                                         cur_plan_config=job_to_plan_config[job],
                                         optimize_all_fn=optimize_all_fn_by_job(job),
                                        )
      # self.sch.log(f"eval_best_effective_speed_up: \n{job} > {effective_speed_up*100:2f}%\n{new_plan_config if effective_speed_up > 0 else None}\n", level=logging.DEBUG)
      eval_results[job] = (new_plan_config, effective_speed_up)
      
    if not SUIKA_POLICY_MULTITHREAD:
      for job in running_jobs:
        eval_worker(job, acc_list)
    else:
      import threading
      threads = []
      for job in running_jobs:
        thread = threading.Thread(target=eval_worker, args=(job, acc_list))
        threads.append(thread)
        thread.start()
      
      for thread in threads:
        thread.join()

    # find best effective speed up
    for job in running_jobs:
      new_plan_config, effective_speed_up = eval_results[job]
      if new_plan_config is None:
        continue
      
      if effective_speed_up > best_effective_speed_up and \
        new_plan_config.world_size() <= job.usr_plan_config.world_size() * SUIKA_POLICY_HARD_SCALE_LIMIT:
        
        best = (job, new_plan_config)
        best_effective_speed_up = effective_speed_up

    if best_effective_speed_up >= threshold and best is not None:
      # set rank_mesh for multi round
      job, new_plan_config = best
      cur_plan_config = job_to_plan_config[job]
      assert cur_plan_config.rank_mesh is not None, f"rank mesh of {job} cur_plan_config is not set"
      new_rank_mesh = convert_to_rank_mesh_respectfully(acc_mesh=new_plan_config.acc_mesh,
                                                        ref_acc_mesh=cur_plan_config.acc_mesh,
                                                        ref_rank_mesh=cur_plan_config.rank_mesh)
      new_plan_config.rank_mesh = new_rank_mesh
      return best
    else:
      return None    
  
  def _tentative_preempt(self) -> bool:
    running_jobs: List[Job] = self.sch._get_running_jobs(locking=False)
    from Suika.cluster.estimator import eval_effective_speed_up
    
    v_rm = copy.deepcopy(self.sch.rm)
    ppcs = {
      job: copy.deepcopy(ppc) for job, ppc in self.prev_configs.items() # job cannot be deep copied due to socket and locks
    }
    
    current_threshold = self._get_threshold(v_rm)
    max_worst_esu = 0.0

    def preempt_worst():
      nonlocal max_worst_esu
      decision = None
      worst_esu = 10
      for job in running_jobs:
        ppc = ppcs.get(job, [])
        # self.sch.log(f">> preempt_worst: {job} len({len(ppc)}), {[pc.world_size() for pc in ppc]}", level=logging.DEBUG)
        if len(ppc) < 2:
          continue
        esu = eval_effective_speed_up(ppc[-2], ppc[-1], optim=job.job_optimizer, fallback=0.99)
        # self.sch.log(f">> preempt_worst: {job} ESU {esu}", level=logging.DEBUG)
        if esu < worst_esu:
          decision = job, ppc[-2], ppc[-1]
          worst_esu = esu
      self.sch.log(f"preempt_worst: {decision[0] if decision is not None else None} {worst_esu*100:.1f}%", level=logging.DEBUG)
      max_worst_esu = max(max_worst_esu, worst_esu)
      return decision, worst_esu
      
    job_to_plan_config: Dict[Job, PlanConfig] = {}
    
    while True:
      decision, worst_esu = preempt_worst()
      if decision is None:
        self.sch.log(f"tentative_preempt: give up after all possible evictions", level=logging.DEBUG)
        return False
      else:
        job, pc1, pc2 = decision
        if not v_rm.raw_free(pc2.acc_mesh, strict=True):
          assert False, f"tentative preempt: cannot raw free {job} {pc2}; check your code"
        if not v_rm.raw_allocate(pc1.acc_mesh, strict=True):
          assert False, f"tentative preempt: cannot raw allocate {job} {pc1}; check your code"

        ppcs[job].pop(-1)
      
      job_to_plan_config[job] = pc1
      
      # always try to schedule waiting job
      sch_decision = self._schedule(rm=v_rm) 
      if sch_decision is not None:
        self.sch.log(f"tentative_preempt: _schedule succeeded {sch_decision}", level=logging.DEBUG)
        break
      elif max_worst_esu < current_threshold and SUIKA_POLICY_RECONFIG_PREEMPT:
        # try to reconfig if worst_esu < current_threshold and waiting job cannot be scheduled
        sch_decision = self._reconfig(rm=v_rm, threshold=current_threshold, job_to_plan_config=job_to_plan_config)
        if sch_decision is not None:
          self.sch.log(f"tentative_preempt: _reconfig succeeded {sch_decision}", level=logging.DEBUG)
          break

    # avoid preemption that does not intersect with sch_decision
    sch_job, sch_pc  = sch_decision
    used_accs = set(sch_pc.acc_list())
    self.cached_decisions = []
    for job, new_pc in job_to_plan_config.items():
      cur_pc = self.prev_configs[job][-1]
      cur_accs = cur_pc.acc_list()
      if any([cur_acc in used_accs for cur_acc in cur_accs]):
        self.sch.log(f"tentative_preempt intersected with {job}, preemption cached",  level=logging.DEBUG)
        self.cached_decisions.append((job, new_pc))
      else:
        self.sch.log(f"tentative_preempt not intersected with {job}, preemption discarded",  level=logging.DEBUG)

    if sch_job.status != JobStatus.WAITING:
      # if reconfig, append to cached decisions
      self.cached_decisions.append(sch_decision)

    self.sch.log(f"tentative_preempt() selected: {sch_job}")
    return True
      
  def _discard_cache(self):
    self.cached_decisions = []
    self.cached_chain_decisions = {}
   
  def _get_threshold(self, rm) -> float:
    # active_jobs = self.sch._get_active_jobs()
    active_jobs = self.sch._get_running_jobs()
    pessmistic_n_accs = sum(
      [job.usr_plan_config.world_size() for job in active_jobs]
    )
    total_n_acc = rm.get_total_acc_num()
    pessmistic_util = pessmistic_n_accs / total_n_acc if total_n_acc > 0 else 1.0
    return math.pow(pessmistic_util, SUIKA_POLICY_TH_EXP)
   
"""
  Metis Policy  
"""
metis_time_budget = int(os.getenv("METIS_TIME_BUDGET", "60"))

class PolicyMetis(PolicySuika):
  def __init__(self, scheduler: 'Scheduler'):
    super().__init__(scheduler=scheduler)
    self.sch.log("PolicyMetis initialized")
  
  def _reconfig(self, *args, rm=None, **kwargs) -> Union[None, Tuple[Job, PlanConfig]]:
    from Suika.external.Metis.suika_interface import Metis_optimize_all
    
    def optimize_all_fn_by_job(job):
      def optimize_all_fn(plan_config, new_acc_units):
        return Metis_optimize_all(job=job, cur_plan_config=plan_config, new_acc_units=new_acc_units, time_budget=metis_time_budget, rm=rm)
      return optimize_all_fn
    
    return super()._reconfig(*args, optimize_all_fn_by_job=optimize_all_fn_by_job, rm=rm, **kwargs)


"""
  DP Policy
  apply the same effective_speed_up criteria as Suika, but optimize all only consider symmetric DP scaling.
"""
class PolicyDP(PolicySuika):
  
  prev_configs: Dict[Job, PlanConfig]
  
  def __init__(self, scheduler: 'Scheduler'):
    super().__init__(scheduler=scheduler)
    self.sch.log("PolicyDP initialized")
    self.prev_configs = {}
  
  def _reconfig(self, *args, rm=None, **kwargs) -> Union[None, Tuple[Job, PlanConfig]]:
    from Suika.external.DP.suika_interface import DP_optimize_all
    
    minimal_scaling = False
    def optimize_all_fn_by_job(job):
      def optimize_all_fn(plan_config, new_acc_units):
        return DP_optimize_all(job=job, cur_plan_config=plan_config, new_acc_units=new_acc_units, minimal_scaling=minimal_scaling, rm=rm)
      return optimize_all_fn
    
    return super()._reconfig(*args, optimize_all_fn_by_job=optimize_all_fn_by_job, rm=rm, **kwargs)

"""
  Rubick Policy
  This policy is a wrapper for Rubick's policy implementation.
"""
class PolicyRubick(Policy):
  
  impl: 'PolicyRubickImpl'
  
  def __init__(self, scheduler: 'Scheduler'):
    super().__init__(scheduler=scheduler)
    self.sch.log("PolicyRubick initialized")
    from Suika.external.Rubick.suika_interface import PolicyRubickImpl
    self.impl = PolicyRubickImpl(scheduler)
  
  def submit(self, job: Job):
    self.impl.submit(job)
  
  def schedule(self) -> Union[Tuple[Job, PlanConfig], None]:
    return self.impl.schedule()
  
  def launch_failure_callback(self, job, plan_config):
    self._discard_cache()
  
  def _discard_cache(self):
    return self.impl._discard_cache()

"""
  Sia Policy
  This policy is a wrapper for Sia's policy implementation.
"""
class PolicySia(Policy):
  
  impl: 'PolicySiaImpl'
  
  def __init__(self, scheduler: 'Scheduler'):
    super().__init__(scheduler=scheduler)
    self.sch.log("PolicySia initialized")
    from Suika.external.Sia.suika_interface import PolicySiaImpl
    self.impl = PolicySiaImpl(scheduler)
  
  def submit(self, job: Job):
    self.impl.submit(job)
  
  def schedule(self) -> Union[Tuple[Job, PlanConfig], None]:
    return self.impl.schedule()
  
  def launch_failure_callback(self, job, plan_config):
    return self._discard_cache()
  
  def _discard_cache(self):
    return self.impl._discard_cache()
  

"""
Get policy with policy name
"""
def get_policy(policy_name: str, scheduler: "Scheduler") -> Policy:
  policy_name = policy_name.lower()
  if policy_name == "default":
    policy_name = "suika"
    
  if policy_name == "suika":
    return PolicySuika(scheduler)
  elif policy_name == "fcfs":
    return PolicyFCFS(scheduler, allow_reconfig=False)
  elif policy_name == "sjf":
    return PolicySJF(scheduler, allow_reconfig=False)
  elif policy_name == "dp":
    return PolicyDP(scheduler)
  elif policy_name == "metis":
    return PolicyMetis(scheduler)
  elif policy_name == "rubick":
    return PolicyRubick(scheduler)
  elif policy_name == "sia":
    return PolicySia(scheduler)
  elif policy_name == "dummy":
    return PolicyDummy(scheduler)
  else:
    assert False, f"unsupported policy type {policy_name}"
    
def get_policy_name(policy_instance: Policy) -> str:
  if isinstance(policy_instance, PolicySuika):
    if isinstance(policy_instance, PolicyDP):
      return "dp"
    if isinstance(policy_instance, PolicyMetis):
      return "metis"
    return "suika"
  elif isinstance(policy_instance, PolicyFCFS):
    return "fcfs"
  elif isinstance(policy_instance, PolicySJF):
    return "sjf"
  elif isinstance(policy_instance, PolicyRubick):
    return "rubick"
  elif isinstance(policy_instance, PolicySia):
    return "sia"
  elif isinstance(policy_instance, PolicyDummy):
    return "dummy"
  else:
    return "unknown"