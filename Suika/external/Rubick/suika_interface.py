import copy
from typing import TYPE_CHECKING, Dict, List, Set, Tuple, Union

if TYPE_CHECKING:
  from Suika.cluster.scheduler import Scheduler

from Suika.external.Rubick.performance_model import PerformanceModel
from Suika.external.Rubick.utils import rubick_logger
from Suika.external.Rubick.hyperparam import (
  k_queuing_penalty_threshold,
  k_reconfig_penalty_threshold,
  patch_rubick_non_local_job_slope,
  patch_rubick_queuing_preempt,
  patch_rubick_best_effort,
  perf_curve_min, perf_curve_max
)
from Suika.models.utils import ModelStrategy
from Suika.cluster.job import Job, JobStatus
from Suika.cluster.plan_config import PlanConfig
from Suika.cluster.resource import Accelerator, ResourceManager
  
# Remark:
# - 1. Resource types "num_acc, n_thread, bandwidth" are reduced only to "num_acc"; Rubick artifact does not consider n_thread and bandwidth for 3D jobs, and we follow that.
# - 2. The experiment traces contain no SLO info, so we ignore SLO-related design in Rubick. But the strategy to find a better initial plan is preserved.
  
class PolicyRubickImpl():
  
  """
  Fields in Job.other_context:
  1. "model_strategy": ModelStrategy
    predefined user plan descriptor, and model performance data
    
    must be set at submit
    
  2. "perf_model": PerformanceModel
    performance model of the job
    
    constructed in submit()
    
  3. "min_res": int
    minimum resource that achieves the same throughput as user plan
    
    set at submit()
  """
  
  sch: "Scheduler"
  cached_decisions: List[Union[Tuple[Job, PlanConfig], None]]
  
  cached_performance_model: Dict[str, PerformanceModel] # necessary for simulation, since n_gpu can be thunsands, cannot recompute for each seperate job.
  
  def __init__(self, sch: "Scheduler"):
    self.sch = sch
    self.cached_decisions = []
    self.cached_performance_model = {}
    
  def submit(self, job: Job):
    """
    model_strategy: ModelStrategy must be set in job's other context field
    
    set "perf_model" field in other context
    """
    try:
        model_strategy=job.other_context["model_strategy"]
    except KeyError:
      raise RuntimeError("model_strategy not set in job's other context")
    
    ms: str = str(model_strategy)
    if ms not in self.cached_performance_model:
      self.cached_performance_model[ms] = PerformanceModel(
        model_strategy=model_strategy
      )
      
    job.other_context["perf_model"] = copy.deepcopy(self.cached_performance_model[ms])
    
    rubick_logger.debug(f"submit: {job}")
    self._reset_initial_plan(job)
  
  
  def schedule(self) -> Union[Tuple[Job, PlanConfig], None]:
    """
      if (job, None) is returned, it means the job is preempted.
    """
    
    # Rubick schedule all jobs in a schedule round, while our scheduler query policies for next decision one by one.
    # so we need to cache the decisions.
    if not self.cached_decisions:
      self.optimize()
    
    while self.cached_decisions:
      job, plan_config = self.cached_decisions.pop(0)
      if job.status in [JobStatus.RUNNING, JobStatus.WAITING]:
        action = ("launch" if job.status == JobStatus.WAITING else "reconfig") if plan_config else "preempt"
        rubick_logger.info(f"schedule(): {job}@{action}")
        return (job, plan_config)
      else:
        rubick_logger.warn(f"schedule() failed: {job} not WAITING/RUNNING")
        self._discard_cache()
        break
    
    return None
  
  def _discard_cache(self):
    self.cached_decisions = []
  
  def optimize(self):
    """
      Function Schedule(jobs, cluster) in Section 5.2
      
      job.res, job.placement, job.plan are inherently represented by job.cur_plan_config
      
      free_res is stored in self.sch.rm
    """
    rubick_logger.debug(f"optimize@({self.sch._record_time():.0f})")
    
    # get all waiting and running jobs that may be scheduled
    jobs: List[Job] = self.sch._get_waiting_and_running_jobs(locking=False)
    rubick_logger.debug(f"> unfiltered({len(jobs):2}) {[(job.job_name, str(job.status.name)) for job in jobs]}")
    # fitler out jobs that are not reconfigurable
    any_queuing_too_long = any([self._is_queuing_too_long(job) for job in jobs])
    jobs = [job for job in jobs if self._is_reconfigurable(job, any_queuing_too_long)]
    rubick_logger.debug(f">   filtered({len(jobs):2}) {[(job.job_name, str(job.status.name)) for job in jobs]}")
    # note: jobs here contains no pending jobs, and so will not pending jobs be preempted. the following code ensures correctness by querying free resources from rm, not solely from job_res dict.
    
    job_slopes: Dict[Job, float] = {
      job: self._get_job_slope(job)
      for job in jobs
    }    
    sorted_jobs: List[Job] = sorted(
      jobs,
      key=lambda job: job_slopes[job],
      reverse=True,
    )
    rubick_logger.debug(f"> sorted jobs {[(job.job_name, job_slopes[job]) for job in sorted_jobs]}")
    
    # internal state change need to be traced across schedule_job
    # so deepcopy to avoid changing actual scheduler's state    
    # virtual_rm: resource allocation state
    virtual_rm = copy.deepcopy(self.sch.rm)
    # job_ref: resource provisional state
    job_res: Dict[Job, List[Accelerator]] = {
      job: job.cur_plan_config.acc_list() if job.cur_plan_config else []
      for job in sorted_jobs
    }
    
    # schedule jobs
    new_plan_configs: List[PlanConfig] = []
    for i, job in enumerate(sorted_jobs):
      # can only preempt jobs that are scheduled after this job
      preemptible_jobs = sorted_jobs[i+1:]
      plan_config = self.schedule_job(
        job,
        preemtible_jobs=preemptible_jobs,
        rm=virtual_rm,
        job_res=job_res,
      )
      new_plan_configs.append(plan_config) # plan_config here is already allocated with physical resource
      
    # compare plan_config difference and update cached decisions
    cur_plan_configs = {
      job: job.cur_plan_config for job in sorted_jobs
    }
    new_plan_configs = {
      job: new_plan_config for job, new_plan_config in zip(sorted_jobs, new_plan_configs)
    }
    self.cached_decisions = self._convert_decisions(jobs=sorted_jobs, cur_pcs=cur_plan_configs, new_pcs=new_plan_configs)
    return
      
  def schedule_job(self, 
                   job: Job, 
                   preemtible_jobs: List[Job],
                   rm: ResourceManager,
                   job_res: Dict[Job, List[Accelerator]],
                  ) -> Union[PlanConfig, None]:
    """
      Function ScheduleJob(job, cluster) in Section 5.2
      
      return plan config after rescheduling. 
      
      if None is returned, it means:
        1. if job is already running, it is preempted
        2. if job is not running, it is not scheduled
    """
    def dbg_log(s: str):
      # rubick_logger.debug(f"    - {s}")
      pass
    dbg_log(f"> schedule_job {job}")
    min_res = job.other_context["min_res"]
    steal_record: Dict[Accelerator, Job] = {}
    
    for node in rm.get_node_list():
      dbg_log(f"= node {node.node_id} =")

      node_res = node.get_avail_acc_list()
      job_res[job].extend(node_res)
      
      jobs_on_this_node = self._get_jobs_on_this_node(node.node_id, preemtible_jobs, job_res)
      dbg_log(f"jobs_on_this_node {jobs_on_this_node}")

      while True:
        victim_job = self._get_lowest_slope_over_min_job(jobs_on_this_node, job_res)
        dbg_log(f"victim {victim_job}")
        if victim_job is None or victim_job == job:
          break
        
        cur_job_slope = self._get_job_slope(job, num_acc=len(job_res[job]))
        victim_job_slope = self._get_job_slope(victim_job, num_acc=len(job_res[victim_job]), delta=-1)
        
        dbg_log(f"cur_job_slope {cur_job_slope:.2f} ?> victim_job_slope {victim_job_slope:.2f} | cur_res {len(job_res[job])} <? min_res {min_res}")

        if cur_job_slope > victim_job_slope or len(job_res[job]) < min_res:
          idx = self._find_acc_on_node(node.node_id, job_res[victim_job])
          if idx is None:
            jobs_on_this_node.remove(victim_job)
            continue
          acc = job_res[victim_job].pop(idx)
          dbg_log(f"stolen ({acc.node_id}, {acc.local_id})")
          steal_record[acc] = victim_job
          job_res[job].append(acc)
        else:
          break
        
      
    def finalize_rm_state_and_job_res(plan_config) -> None:
      # helper function before returning
      # 1. return unused stolen res to victim job, and allocate res back
      # 2. allocated used res
      used_res = plan_config.acc_list() if plan_config else []
      unused_res = [acc for acc in job_res[job] if acc not in used_res]
      unused_stolen_res = [acc for acc in unused_res if acc in steal_record]
      
      for acc in unused_stolen_res:
        job_res[steal_record[acc]].append(acc)
        if not rm.raw_allocate(acc, strict=True):
          assert False, f"fail to allcoate {acc} which is stolen by {job} from {steal_record[acc]}"
      job_res[job] = used_res # avoid holding node's free_res that does not belong to any job yet.
      for acc in used_res:
        if not rm.raw_allocate(acc, strict=True):
          assert False, f"fail to allcoate {acc} which is in used by {job}"
      
    # free stolen resources to test allocation
    rm.raw_free(job_res[job], strict=False)
    
    dbg_log(f"provisioned {len(job_res[job])} <=> min {min_res}")
    dbg_log(f"{[(acc.node_id, acc.local_id) for acc in job_res[job]]}")

    if len(job_res[job]) >= min_res:
      perf_model: PerformanceModel = job.other_context["perf_model"]
      cur_accs = job.cur_plan_config.acc_list() if job.cur_plan_config else []
      cur_accs = [acc for acc in cur_accs if acc in job_res[job]] # filter out accelerators that have been stolen by other jobs
      
      # find best plan that can be allocated with placement constraint
      for num_acc in range(len(job_res[job]), min_res - 1, -1):
        strategy = perf_model.perf_curve[num_acc][1]
        if strategy is None:
          break
        v_plan_config = strategy.get_plan_config()
        
        if v_plan_config.world_size() <= len(cur_accs):
          # reuse current gpus first, avoid unnecessary migration
          acc_mesh = rm.allocate(v_plan_config.acc_mesh, virtual_test=True, mask=cur_accs)
          if acc_mesh is None:
            acc_mesh = rm.allocate(v_plan_config.acc_mesh, virtual_test=True, mask=job_res[job])
        else:
          acc_mesh = rm.allocate(v_plan_config.acc_mesh, virtual_test=True, mask=job_res[job])
        
        if acc_mesh is not None:
          p_plan_config = v_plan_config.instantiate(acc_mesh)
          finalize_rm_state_and_job_res(p_plan_config)
          # rubick_logger.info(f"{job} is scheduled with plan {p_plan_config}")
          num_acc = p_plan_config.world_size()
          dbg_log(f"< schedule_job {job}")
          rubick_logger.debug(f" | {job.job_name:15} n_acc={num_acc}")
          return p_plan_config

    dbg_log(f"< schedule_job {job}")
    rubick_logger.debug(f" | {job.job_name:15} n_acc={0}")
    finalize_rm_state_and_job_res(None)
    return None
      
  """
  UTILS
  """
  
  @staticmethod
  def _reset_initial_plan(job: Job):
    """
      find the plan with least resources to achieve the same throughput as user plan
      
      reset user_plan_config 
      set "min_res" field in other context
    """
    
    orig_strategy: ModelStrategy = job.other_context["model_strategy"]
    orig_num_acc = orig_strategy.get_world_size()
    
    perf_model: PerformanceModel = job.other_context["perf_model"]
    orig_th = perf_model.orig_th
    
    for num_acc in range(1, orig_num_acc + 1):
      th, new_strategy = perf_model.perf_curve[num_acc]
      if th >= orig_th:
        # found
        job.usr_plan_config = new_strategy.get_plan_config()
        if patch_rubick_best_effort: # if best effort, minRes is 0
          num_acc = 0
        job.other_context["min_res"] = num_acc
        return

    assert False, f"incorrect perf_curve (not optimal) for {job}, no plan found even with orig_num_acc"
  
  
  def _get_job_slope(self, job: Job, delta: int = 1, num_acc: int = None):
    perf_model: PerformanceModel = job.other_context["perf_model"]
    
    if num_acc is None:
      num_acc = job.cur_plan_config.world_size() if job.cur_plan_config else 0
      
    if self._is_queuing_too_long(job):
      return float("inf")
    
    old_th = perf_model.perf_curve[num_acc][0]

    # if patch rubick non local job slope:
    #   find clostest new_th on the delta direction that differs from old_th
    # else:
    #   find the new_th at exact delta
    if patch_rubick_non_local_job_slope:
      new_th = old_th
      while new_th == old_th and perf_curve_min <= num_acc + delta <= perf_curve_max:
        new_th = perf_model.perf_curve[num_acc + delta][0]
        # rubick_logger.info(f"? {old_th:.2f} vs {new_th:.2f} at {num_acc} + {delta}")
        delta += 1 if (delta > 0) else -1
    else:
      new_th = perf_model.perf_curve[num_acc + delta][0]
    
    return (new_th - old_th) / perf_model.orig_th / delta
  
  def _is_reconfigurable(self, job: Job, any_queuing_too_long: bool) -> bool:
    """
      test if job is reconfigurable against reconfig penalty
    """
    if job.status == JobStatus.WAITING:
      return True

    if job.status == JobStatus.RUNNING and patch_rubick_queuing_preempt and any_queuing_too_long:
      return True

    reconfig_cost = self.sch._get_job_reconfig_cost(job, exclude_first=True) # if we do not exclude first launch cost, following jobs will likely be blocked for hours...
    up_time = self.sch._get_job_up_time(job)
    
    reconfig_pentaly_factor = 1 - reconfig_cost / (up_time + 1e-2) # larger -> smaller reconfig cost
    
    return reconfig_pentaly_factor >= k_reconfig_penalty_threshold

  def _is_queuing_too_long(self, job: Job):
    return job.status == JobStatus.WAITING and \
           not len(self.sch.job_statistics[job]["start_time"]) and \
           self.sch._get_job_age(job) > k_queuing_penalty_threshold
  
  @staticmethod
  def _find_acc_on_node(node_id: int, acc_list: List[Accelerator], assertion = False) -> Union[int, None]:
    for i, acc in enumerate(acc_list):
      if acc.node_id == node_id:
        return i
    
    if assertion:
      assert False, "accelerator not found on node"
    return None
  
  @staticmethod
  def _get_jobs_on_this_node(node_id: int, preemptible_jobs: List[Job], job_res: Dict[Job, List[Accelerator]]) -> List[Job]:
    jobs_on_this_node = []

    for job in preemptible_jobs:
      if PolicyRubickImpl._find_acc_on_node(node_id, job_res[job], assertion=False) is not None:
        jobs_on_this_node.append(job)
      
    return jobs_on_this_node
  
  def _get_lowest_slope_over_min_job(self, preemptible_jobs_on_this_node: List[Job], job_res: Dict[Job, List[Accelerator]]):
    
    lowest_slope_job, lowest_slope = None, float("inf")
    for job in preemptible_jobs_on_this_node:
      if len(job_res[job]) <= job.other_context["min_res"]:
        continue
      slope = self._get_job_slope(job, num_acc=len(job_res[job]), delta=-1)
      if slope < lowest_slope:
        lowest_slope = slope
        lowest_slope_job = job
    
    return lowest_slope_job
  
  def _convert_decisions(self, jobs: List[Job], cur_pcs: Dict[Job, PlanConfig], new_pcs: Dict[Job, PlanConfig]) -> List[Union[Tuple[Job, PlanConfig], None]]:
    # there may be cycles in the allocate / deaclloate dependency graph..., to avoid deadlock, we detect rings, and decouples ring to preempt to 0 gpu and then launch again
    jobs = [job for job in jobs if cur_pcs[job] != new_pcs[job]]
    
    edge: Dict[Job, Set[Job]] = {
      job: set() for job in jobs
    } # a -> b if.f. new_pcs[b] overlaps with cur_pcs[a], a must be eariler than b
    ind: Dict[Job, int] = {
      job: 0 for job in jobs
    }
    
    def pc_overlap(pc1: PlanConfig, pc2: PlanConfig) -> bool:
      acc_list1 = pc1.acc_list() if pc1 else []
      acc_list2 = pc2.acc_list() if pc2 else []
      return set(acc_list1).isdisjoint(set(acc_list2)) is False
    
    for i in jobs:
      for j in jobs:
        if i == j:
          continue
        if pc_overlap(cur_pcs[i], new_pcs[j]):
          edge[i].add(j)
          ind[j] += 1
    
    # topological sort to find the order of jobs, and potential rings
    inf = 10000
    topo_order: Dict[Job, int] = {
      job: inf for job in jobs
    }
    queue: List[Job] = []
    
    for job in jobs:
      if ind[job] == 0:
        queue.append(job)
        topo_order[job] = 0
    
    while queue:
      i = queue.pop(0)
      for j in edge[i]:
        ind[j] -= 1
        if ind[j] == 0:
          queue.append(j)
          topo_order[j] = topo_order[i] + 1
          
    jobs = list(sorted(jobs, key=lambda job: topo_order[job]))
    decisions: List[Union[Tuple[Job, PlanConfig], None]] = []
    ring_decisions_preempt = []
    ring_decisions_launch = []
    
    for job in jobs:
      cur_pc = cur_pcs[job]
      new_pc = new_pcs[job]
      cur_n_acc = cur_pc.world_size() if cur_pc else 0
      new_n_acc = new_pc.world_size() if new_pc else 0
      order = topo_order[job]
      
      if order is not inf:
        rubick_logger.debug(f"< {job.job_name:15}({order}) {cur_n_acc} -> {new_n_acc}")
        decisions.append((job, new_pc))
      else:
        # note that new_n_acc = 0 will never be in a ring
        rubick_logger.debug(f"< {job.job_name:15}(inf) {cur_n_acc} -> 0 -> {new_n_acc}")
        ring_decisions_preempt.append((job, None))
        ring_decisions_launch.append((job, new_pc))
      
    decisions += ring_decisions_preempt + ring_decisions_launch
    return decisions