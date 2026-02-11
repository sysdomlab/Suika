# Schduler process implement. handle cluster management and configuration.
import copy
from enum import Enum
import json
import os
import logging
from logging import Logger
from typing import Any, Callable, List, Optional, Set, Tuple, Union, Dict
import threading
import time
import math

from Suika.cluster.commu import SuikaSocketServer, Scheduler2ExecutorMsg
from Suika.cluster.estimator import JobOptimizer
from Suika.cluster.job import Job, JobStatus, LaunchConfig
from Suika.cluster.plan_config import PlanConfig, convert_to_rank_mesh, convert_to_rank_mesh_respectfully
from Suika.cluster.resource import HostnameMesh, ResourceManager, convert_to_hostname_mesh, parse_cluster_config
from Suika.cluster.policy import get_policy, get_policy_name, Policy
from Suika.device_mesh import RawDeviceMesh3D
from Suika.utils import flatten_mesh


"""
Static Constants
"""
default_scheduler_port_begin = 20811
default_worker_port_begin = 12355
default_schedule_interval = 30
# default_simu_schedule_interval = 30
default_simu_schedule_interval = int(os.getenv("SUIKA_SIMU_SCHEDULE_INTERVAL", str(default_schedule_interval)))
default_ff_time_margin = 30 # margin of time to be preserved for fast forward, in seconds
default_ff_time_min_range = 60 # minimum length of time to be fast forwarded in one jump # in additional to 2 * margin
default_ff_time_max_range = 200 # maximum length of time to be fast forwarded in one jump; avoid skip too much for time-related scheduling (e.g., reconfig penalty reduces over time even without reset event)
def set_ff_time_max_range(value: float):
  global default_ff_time_max_range
  default_ff_time_max_range = value
  # hack: this value is only used within this module, so it's fine.

"""
The running instance of the Scheduler.
Actual scheduling policy and resource management are handled by the Policy and ResourceManager classes respectively.
"""
class Scheduler:
  
  """
  Members
  """
  policy: Policy
  rm: ResourceManager
    
  # scheduler communication
  scheduler_addr: str
  scheduler_port_begin: int
  job_port_usage: Set[int]
  
  # work info
  hostname_list: List[str]
  worker_port_begin: int
  worker_port_usage: Dict[int, Set[int]]
  
  # job info
  job_id_counter: int
  jobs: Dict[int, Job]
  active_jobs: Set[int] # a set of active jobs' ids; active means, waiting, pending or running.
  
  # executor server
  executor_port: int
  executor_server: SuikaSocketServer
  
  # logger
  logger: Logger
  
  # guard
  global_lock: threading.Lock
  guard_thread: Optional[threading.Thread] = None
  is_running: bool
  
  # statistics
  time_base: float
  job_statistics: Dict[Job, Dict[str, float]]

  # ff  
  ff_offset: float 
  last_successful_schedule_time: float
  last_ff_wallclock_time: float
  oracle_submit_time: List[float]

  # other scheduling hyperparams
  schedule_interval: int
  
  """
  Methods
  """

  """
  Initialize Scheduler from configuration file or dictionary.
  
  Args:
    config: config dictionary or file path to the config file. (default: "~/.suika.json")
    policy: str (Optional), scheduling policy, overrides policy provided in config.
  
  Config format:
    1. accelerator: 
      dict[
        key=type_name, 
        value={TFOLPS, memory}
      ] 
    2. node: 
      dict[
        key=hostname, 
        value={acc_type, node_id, acc_num}
      ]
    3. network:
      dict[
        key=
          acc_type(str) - intra bandwidth
          or 
          acc_type,acc_type (str, split by ',', no space) - inter bandwidth
        value=bandwidth
      ]
    
    4. policy: 
      str (Optional), can be overridden by __init__ parameter 'policy'
    
    5. scheduler_addr:
      str, address of the scheduler, used for job process to connect to job manager
      
    6. scheduler_port_begin:
      int (Optional), the begining of available port of the scheduler, used for job process to connect to job manager
      
      default to 40402 
      
      the exact scheduler_port_begin is reserved as executor port, actual first port is scheduler_port_begin + 1
            
    7. worker_port_begin:
      int (Optional), the begining of available port of the worker, used for master_port
      
      default to 12355
      
      remark: each job will be assigned a master_port s.t. port % 2 = 0
              and reconfiguration should use master_port and master_port | 1 alternatively (this is managed by scheduler automatically)
    
  Example config:
    {
        "accelerator": {
            "A100": {
                "TFLOPS": 312.0,
                "memory": 40000
            },
            "V100": {
                "TFLOPS": 125.0,
                "memory": 32000
            }
        },
        "node": {
            "node0": {
                "acc_type": "A100",
                "node_id": 0,
                "acc_num": 4
            },
            "node1": {
                "acc_type": "V100",
                "node_id": 1,
                "acc_num": 8
            }
        },
        "network": {
            "A100": 300000,
            "V100": 64000,
            "V100,A100": 25000,
            "A100,A100": 25000,
            "V100,V100": 25000
        }
        "policy": "default",
        "scheduler_addr": "localhost"
    }
    
    Remark: callback is deprecated. scheduler and job communicate inside the same process, no more socket
  """
  
  def __init__(self, 
               config: Union[os.PathLike, Dict] = "~/.suika.json", 
               policy: Optional[str] = None, 
               ff_enable: bool = True,
               schedule_interval: int = default_schedule_interval,
               logger: Optional[Logger] = None,
               start_after_init: bool = True
               ):
    """
    Initialize the scheduler instance.
    """
    # loggers
    self.logger = logger # move logger initialization ahead to allow .log call during __init__
    
    if not isinstance(config, dict):
      # read config from file
      with open(os.path.expanduser(config), 'r') as f:
        kwargs = json.load(f)
    else:
      kwargs = config
      
    # resource manager & hostname_list      
    self.rm, hostname_list, cluster_config = parse_cluster_config(kwargs)
    self.log(f"cluster_config:\n{cluster_config}")
    
    # other scheduling hyperparameters (set before policy, as some policy requires it)
    self.schedule_interval = schedule_interval

    # policy
    policy = kwargs.get("policy", "default") if policy is None else policy
    self.log(f"policy: {policy}")
    self.policy = get_policy(policy_name=policy, scheduler=self)
    
    # scheduler endpoint
    scheduler_addr = kwargs.get("scheduler_addr", None)
    executor_port = kwargs.get("scheduler_port_begin", default_scheduler_port_begin)
    scheduler_port_begin = executor_port + 1
    assert scheduler_addr is not None, "scheduler_addr is required"
    
    self.scheduler_addr = str(scheduler_addr)
    self.scheduler_port_begin = int(scheduler_port_begin)
    self.job_port_usage = set()
    
    # worker
    self.hostname_list = hostname_list
    self.worker_port_begin = kwargs.get("worker_port_begin", default_worker_port_begin)
    self.worker_port_usage = { hostname:set() for hostname in self.hostname_list}
    
    # job
    self.job_id_counter = 0
    self.jobs = {}
    self.active_jobs = set()
    
    # executor
    self.executor_port = executor_port
    self.executor_server = SuikaSocketServer(f"{scheduler_addr}:{executor_port}", logger = logger)
    
    # logger
    # self.logger = logger
    
    # guard
    self.global_lock = threading.Lock()
    self.guard_thread = threading.Thread(target=self.guard, daemon=True)
    self.is_running = False

    # statistics
    self.time_base = 0.0
    self.job_statistics = {}
    
    # ff
    self.ff_enable = ff_enable
    self.ff_offset = 0.0
    self.last_successful_schedule_time = 0.0
    self.last_ff_wallclock_time = 0.0
    self.oracle_submit_time = []

    # other scheduling hyperparameters (moved above)
    # pass

    if start_after_init:
      self.start()
  
  """
  User Intreface
  """

  def status(self, job: Union[Job, int]) -> JobStatus:
    """
    Get the status of the job.
    """
    if isinstance(job, int):
      if job not in self.jobs:
        return JobStatus.UNKNOWN
      job = self.jobs[job]
      
    return job.status
  
  def register_submit_time(self, submit_time: float):
    # register submit time in advance for fast forward
    self.oracle_submit_time.append(submit_time)
    return 
    
  def submit(self, job: Job) -> int: # return assigned job id
    """
    Submit the job to the scheduler.
    """
    self.global_lock.acquire()
        
    job.job_id = self.job_id_counter
    self.job_id_counter += 1
    
    job.status = JobStatus.WAITING
    job.scheduler_timer = self._record_time
    job.record_iter_time = lambda iter_time: self._record_job_iter_time(job, iter_time)
    
    # update scheduler state
    self.policy.submit(job)
    self.jobs[job.job_id] = job
    # add to active jobs
    self.active_jobs.add(job.job_id)
    
    
    self.log(f"submit(), {job}")
    self._record_job_submit(job)
    
    self.global_lock.release()
    return job.job_id
  
  def cancel(self, job: Union[Job, int]) -> None:
    """
    Cancel the job.
    """
    self.global_lock.acquire()
    
    if isinstance(job, int):
      job = self.jobs.get(job, None)
      
    job: Job
    if job is not None and job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
      job._handle_cancelled()
      # regardless, release job should be handeled by guard
      
      self.log(f"cancel(), {job}")
    else:
      if job is None:
        self.log(f"cancel() failed, job with job_id = {job} not found", level=logging.CRITICAL)
      else:
        self.log(f"cancel() failed, {job} is not in PENDING or RUNNING state", level=logging.CRITICAL)
      
    self.global_lock.release()
  
  def start(self):
    """
    Start the scheduler.
    """
    self.global_lock.acquire()
    
    if self.is_running:
      self.log("start() failed, scheduler is already running", level=logging.WARNING)
      return
    
    self.is_running = True
    self.time_base = time.time()
    # executor server is started in guard
    self.guard_thread.start()
  
    self.global_lock.release()
    
    self.log("start()")
  
  def stop(self, wait_for_guard = True):
    """
    Exit the scheduler.
    """
    if wait_for_guard:
      self.global_lock.acquire()
    
    if not self.is_running:
      self.log("stop() failed, scheduler is not running", level=logging.WARNING)
      return
    
    self.is_running = False
    
    self.executor_server.broadcast(
      (Scheduler2ExecutorMsg.EXIT, None)
    )
    
    if self.guard_thread is not None and wait_for_guard:
      self.guard_thread.join()
    
    self.executor_server.stop()
    
    if wait_for_guard:
      self.global_lock.release()
    
    self.log("stop()")
  
  def get_job_statistics(self) -> Dict[str, Any]:

    for job in self.job_statistics:
      # remove lock related unpicklable items
      job.reconfig_thread = None
      job.guard_thread = None
      job.guard_lock = None
      job.job_server = None
      job.scheduler_timer = None
      job.record_iter_time = None
      job.reconfig_start_timers = []
      job.reconfig_stop_timers = []
    
    return self.job_statistics
  
  def join(self):

    if not self.is_running:
      return
    
    while len(self.active_jobs):
      self.log(f"join(): {len(self.active_jobs)} active job(s)")
      time.sleep(10)

    self.stop(wait_for_guard=True)
    self.log("join()")

  """
  Internal Interface
  """
        
  def schedule(self) -> Union[int, None]:
    """
    Schedule the next job and launch/reconfig it.
    
    return scheduled job_id or None
    """
    self.global_lock.acquire()
    
    result = None
    retry = 3
    while retry:
      # select next job to reconfigure / launch
      # WARNING: policy should NOT change resource manager's state with allocation
      # self.launch will do that
      decision = self.policy.schedule()
      
      if decision is not None:
        # schedule available
        job, plan_config = decision
        self.log(f"schedule() selected, {job}")
                
        # attempts to launch job
        if self.launch(job, plan_config):
          # succeded, end this schedule round
          result = job.job_id
          self.last_successful_schedule_time = self._record_time()

          self.policy.launch_success_callback(job, plan_config)
          break
        else:
          # failed
          self.log(f"schedule() failed, {job}")

          self.policy.launch_failure_callback(job, plan_config)

          if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            # case 1: job failed or completed during scheduling, launching, or reconfiguring
            self.log(f"schedule() failed, {job} finished during launch", level=logging.DEBUG)
            # a. if job.cur_plan_config set, then new_plan_config = cur_plan_config, nothing happens
            # b. if job.cur_plan_config not set, then this ensure that, the rm's state is restored to before launch
            #     given that release() relies on job.cur_plan_config, this is necessary.
            self.partial_release(job, new_plan_config=plan_config)
            pass # guard will release the job later
          else:
            # case 2: job is still pending (last reconfig not complete)
            # For Suika, current implement ensures that only WAITING/RUNING jobs are scheduled, this should not happen (unless in rare cases where some worker complete during reconfig, but not the entire job)
            # For Sia/Rubick, this may happen, and requires to discrd cached decesions
            self.log(f"schedule() failed, {job} last reconfig not complete or launch failed", level=logging.WARNING)
            self.partial_release(job, new_plan_config=plan_config)
            # self.partial_release(job, plan_config)
            pass # launch returns immediately if job is pending, no resource changes happen yet. so no need to restore rm state with partial release
            
          retry -= 1
      else:
        self.log(f"schedule() idled")
        break # no more valid schedule
      
    self.global_lock.release()
    
    return result
  
  def release(self, job: Job, locking: bool = True) -> bool:
    """
    Release resources occupied by the job upon failure or completion
    
    return True if succeeded.
    
    note that preempt() does not go through release(), but rely on its own rm state update.
    """
    if locking:
      self.global_lock.acquire()
    
    self.log(f"release(), {job}\n{job.cur_plan_config}", level=logging.DEBUG)

    success = job is not None and job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
    
    if success:
      # free occupied resources
      cur_acc_mesh = job.cur_plan_config.acc_mesh if job.cur_plan_config is not None else []
      success = self.rm.raw_free(cur_acc_mesh, strict=True)
      
      # remove from active_jobs
      self.active_jobs.remove(job.job_id)

      job.job_server.stop()
      self._free_job_port(job.job_server.port)
      
      if job.launch_config is not None:
        self._free_master_port(job.launch_config.master_addr, job.launch_config.master_port)
    
    if success:
      self.log(f"release(), {job}")
    else:
      self.log(f"release() failed, {job}", level=logging.CRITICAL)
    
    if locking:
      self.global_lock.release()
      
    self._record_job_end(job)
    self.log(f"release() USAGE after - {self.rm.get_node_usage_dict()}", level=logging.DEBUG)
    return success
  
  def partial_release(self, job: Job, new_plan_config: PlanConfig):
    """
    Release resource belongs to new_plan_config but not job.cur_plan_config
    when reconfiguration is rejected, but current job is still running.
    
    note that is exclude >= new_plan_config.acc_mesh, nothing happens.
    """
    new_acc_mesh = getattr(new_plan_config, "acc_mesh", [])
    old_acc_mesh = getattr(job.cur_plan_config, "acc_mesh", [])
    if not self.rm.raw_free(new_acc_mesh, exclude=old_acc_mesh, strict=True):
      # may fail, but still fine, if launch is rejected immediately and job.cur_plan_config hasn't be set yet.
      self.log(f"partial_release() failed, cannot raw_free {new_acc_mesh} - {old_acc_mesh}", level=logging.WARNING)
    if not self.rm.raw_allocate(old_acc_mesh, exclude=new_acc_mesh, strict=True):
      self.log(f"partial_release() failed, cannot raw_allocate {old_acc_mesh} - {new_acc_mesh}", level=logging.WARNING)
    self.log(f"partial_release(), {job}")

  def launch(self, job: Job, plan_config: PlanConfig, **kwargs) -> bool:
    
    """
    Launch or reconfig a job (depends on JobStatus)
    return True if succeeded.
    
    assume global_lock is held by caller (i.e., self.schedule)
    
    **kwargs will be passed to Job.launch and reconfig. primiary use is to set "checkpoint_resume"
    """

    if job.job_optimizer is not None: # record debug info
      peak_memory_usage = job.job_optimizer.est.feasible(plan_config, output_peak_usage=True)
      estimated_iter_time = job.job_optimizer.est.estimate(plan_config, test_feasbile=False)
    else:
      peak_memory_usage, estimated_iter_time = -1.0, -1.0
    self.log(f"launch() init, {job}\n{plan_config} with iter_time {estimated_iter_time:.3f} / memory {peak_memory_usage * 100:.1f}%", level=logging.DEBUG)
    self.log(f"launch() USAGE before - {self.rm.get_node_usage_dict()}", level=logging.DEBUG)
    
    if job.status not in [JobStatus.WAITING, JobStatus.PENDING, JobStatus.RUNNING]:
      self.log(f"launch() failed, {job} is not in WAITING, PENDING or RUNNING state", level=logging.CRITICAL)
      return False
    
    if job.status == JobStatus.PENDING:
      self.log(f"launch() failed, {job} is still pending, cannot reconfig", level=logging.WARNING)
      return False
    
    acc_mesh = getattr(plan_config, "acc_mesh", [])
    
    # allocate resources, assume acc_mesh is valid and correct
    old_acc_mesh =  getattr(job.cur_plan_config, "acc_mesh", [])
    if not self.rm.raw_allocate(acc_mesh, exclude=old_acc_mesh, strict=True):
      self.log(f"launch() failed, {job} cannot allocate resources; FATAL: this should not happen, check your policy impl", level=logging.FATAL)
      return False
    if not self.rm.raw_free(old_acc_mesh, exclude=acc_mesh, strict=True):
      self.log(f"launch() failed, {job} cannot free resources; FATAL: this should not happen, check your policy impl", level=logging.FATAL)
      return False

    self._record_job_plan_config(job, plan_config)
    self._record_job_reconfig_init(job)
    self._record_job_iter_time(job, -1.0)

    # preempt (special case)
    if plan_config is None:
      return self.preempt(job, **kwargs)
    
    is_reconfig = job.status == JobStatus.RUNNING

    hostname_mesh: HostnameMesh = convert_to_hostname_mesh(acc_mesh, self.rm)
    if not is_reconfig: # launch
      
      rank_mesh = RawDeviceMesh3D(convert_to_rank_mesh(acc_mesh))
      
      lc = LaunchConfig()
      
      rank0_i, rank0_j, rank0_k = rank_mesh.get_coord(0)
      lc.master_addr = hostname_mesh[rank0_i][rank0_j][rank0_k]
      lc.master_port = self._get_master_port(lc.master_addr)
      lc.job_endpoint = f"{self.scheduler_addr}:{self._get_job_port()}"
      
    else: # reconfiguration
      
      lc = copy.deepcopy(job.launch_config)
      if "checkpoint_resume" not in kwargs: # standard Suika migration
        # rank_mesh = RawDeviceMesh3D(convert_to_rank_mesh_respectfully(acc_mesh, 
        #                                                             ref_acc_mesh=job.cur_plan_config.acc_mesh, 
        #                                                             ref_rank_mesh=job.launch_config.rank_mesh)
        #                           )
        # master_addr : unchanged
        # lc.master_port ^= 1 # there appears to be resource leak. for experiment purpose, we never reuse a port as a temporary workaround
        rank_mesh = plan_config.rank_mesh # bug: if we use multiround, compressing multiround reconifg rank change may be wrong, leave this change to PolicySuika.schedule
        lc.master_port = self._get_master_port(lc.master_addr)
      
      else: # checkpoint resume where two plans' placement may not fully overlap, and respectful rank conversion is wrong (and unnecessary). 
        # also, master_addr and port may change due to switch of rank 0
        rank_mesh = RawDeviceMesh3D(convert_to_rank_mesh(acc_mesh))
        self._free_master_port(lc.master_addr, lc.master_port)

        rank0_i, rank0_j, rank0_k = rank_mesh.get_coord(0)
        lc.master_addr = hostname_mesh[rank0_i][rank0_j][rank0_k]
        lc.master_port = self._get_master_port(lc.master_addr)
        
      # job_endpoint : unchanged


    lc.world_size = len(flatten_mesh(acc_mesh))
      
    lc.acc_mesh = acc_mesh
    lc.rank_mesh = rank_mesh
    plan_config.rank_mesh = rank_mesh # set rank_mesh
    lc.hostname_mesh = hostname_mesh
    
    
    def record_start():
      self._record_job_reconfig_start(job)
    def record_stop():
      self._record_job_reconfig_stop(job)
    
    if not is_reconfig:
      # launch the job

      success = job.launch(plan_config = plan_config,
                           launch_config=lc, 
                           rm = self.rm, 
                           executor_server=self.executor_server,
                           logger = self._get_job_logger(job),
                           reconfig_start_timer=record_start,
                           reconfig_stop_timer=record_stop,
                           **kwargs,
                           )
      self._record_job_start(job)
      
    else:
      # issue reconfig
      
      # launch blocking is at job.reconfig side if needed
      success = job.reconfig(plan_config=plan_config, 
                             launch_config=lc,
                             executor_server=self.executor_server,
                             reconfig_start_timer=record_start,
                             reconfig_stop_timer=record_stop,
                             **kwargs,
                            )

    self.log(f"launch(){'' if success else ' failed'}, {job}@{'reconfig' if is_reconfig else 'launch'}")
    self.log(f"launch() USAGE after - {self.rm.get_node_usage_dict()}", level=logging.DEBUG)

    return success
  
  def preempt(self, job: Job, **kwargs) -> bool:
    if job.status != JobStatus.RUNNING:
      self.log(f"launch() preempt failed, {job} is not in RUNNING state", level=logging.CRITICAL)
      return False
    
    def record_start():
      self._record_job_reconfig_start(job)
    def record_stop():
      # self._record_job_reconfig_stop(job)
      pass
      
    # launch blocking is at job.reconfig side if needed
    success = job.reconfig(
      plan_config=None, 
      launch_config=None,
      executor_server=self.executor_server,
      reconfig_start_timer=record_start,
      # reconfig_stop_timer=record_stop,
      **kwargs,
    ) 
    
    if success:
      self.policy.submit(job)
      self._record_job_reconfig_stop(job)
      self._record_job_end(job)
    
    self.log(f"launch(){'' if success else ' failed'}, {job}@preempt")
    self.log(f"launch() USAGE after - {self.rm.get_node_usage_dict()}", level=logging.DEBUG)
    
    return success

  def fast_forward(self) -> bool:
    if not self.ff_enable:
      return False
    
    with self.global_lock:
      start_time = time.time()

      ## no need to check, we always call ff after schedule
      ## time to last ff
      # now = time.time() # wallclock time
      # if now - self.last_ff_wallclock_time < default_ff_time_margin:
      #   return False
      
      # time to last successful schedule
      now = self._record_time() # virtual time
      if now - self.last_successful_schedule_time < default_ff_time_margin:
        return False
      
      # time to closest submit time
      assert len(self.oracle_submit_time) > 0, "oracle_submit_time empty when ff is enabled; did you forget to call Scheduler.register_submit_time?"
      
      if min([
        abs(submit_time - now) for submit_time in self.oracle_submit_time
      ]) < default_ff_time_margin:
        return False
      
      # all active jobs are ready for fast forward
      active_jobs = self._get_active_jobs()
      all_ready_for_ff = all(job.ready_for_ff() for job in active_jobs)
      if not all_ready_for_ff:
        return False
      
      running_jobs = self._get_running_jobs()
      reset_event_times = [
        job.get_ff_completion_time() for job in running_jobs
      ] + [
        submit_time for submit_time in self.oracle_submit_time if submit_time >= now
      ]

      if not reset_event_times:
        return False
      # self.log(f"next_reset_event_times: {reset_event_times[:len(running_jobs) + 1]}")
      
      next_reset_event_time = min(reset_event_times)
      target_ff_time = next_reset_event_time - default_ff_time_margin
      target_ff_time = min(target_ff_time, now + default_ff_time_max_range)
      
      if target_ff_time - now < default_ff_time_min_range:
        return False
      
      # all checks passed, issue fast forward
      self.log(f"fast_forward(): from {now:.0f} to {target_ff_time:.0f}, skipping {target_ff_time - now:.0f} seconds", level=logging.INFO)
      self.log(f"fast_forward(): {self._record_time() / (self._record_time() - self.ff_offset + 1e-3):.1f}x speed up", level=logging.INFO)
      
      self.last_ff_wallclock_time = time.time()
      
      for job in running_jobs:
        job.fast_forward(target_ff_time)
      
      self.ff_offset += (target_ff_time - now) # move ff_offset change after fast_forward, when ff update may still be possible, and wrong scheduler time is recorded

      end_time = time.time()
      # self.log(f"fast_forward() takes {end_time - start_time:.2f} wall clock seconds", level=logging.DEBUG)
      return True

  
  def log(self, msg: str, level: int = logging.INFO) -> None:
    """
    Log the message to the logger.
    """
    if self.logger is not None:
      if getattr(self, "is_running", False):
        msg = f"[scheduler]@({self._record_time():.0f}): {msg}"
      else:
        msg = f"[scheduler]: {msg}"
      self.logger.log(level=level, msg=msg)
  
    
  def guard(self) -> None:
    """
      Guard process to poll job's state update. (so that we do not need socket btw job and scheduler, which is rediculous).
      
      Note that self.schedule is completely sequential from other processing in guard, it is thread-safe to release a job in self.schedule.
    """
    guard_sleep_interval = 0.1
        
    last_schedule_time = self._record_time()
    prev_active_jobs = set()
    prev_job_status = {}
    
    # wait for executor to connect
    with self.global_lock:
      self.executor_server.start()
      self.log("waiting for executor to connect")
      node_id_list = list(self.rm.cluster_node_table.keys())
      for node_id in node_id_list:
        if not self.executor_server.wait_for(node_id, timeout=30):
          self.log(f"executor {node_id} did not connect within timeout.", level=logging.ERROR)
          return
        else:
          self.log(f"executor {node_id} connected")
      self.log("all executors connected")
    
    while self.is_running:

      with self.global_lock:
        active_jobs = self.active_jobs.copy()
            
      for job_id in active_jobs:
        job = self.jobs[job_id]
        if job.status in [JobStatus.WAITING, JobStatus.PENDING, JobStatus.RUNNING]:
          
          if prev_job_status.get(job_id, None) == JobStatus.PENDING and job.status == JobStatus.RUNNING:
            # from PENDING to RUNING
            try:
              stats = self.job_statistics[job]
              pending_time = stats["reconfig_stop_time"][-1] - stats["reconfig_init_time"][-1]
              blocking_time = stats["reconfig_stop_time"][-1] - stats["reconfig_start_time"][-1]
            except Exception as e:
              self.log(f"failed to get pending time: {e}; this should not happen", level=logging.WARNING)
              pending_time = -1
              blocking_time = -1
            self.log(f"guard(), {job} running (pending {pending_time:.0f}s / blocking {blocking_time:.0f}s)")
          
        elif job.status in [JobStatus.COMPLETED, JobStatus.CANCELLED, JobStatus.FAILED]:
          self.log(f"guard(), {job} finished")
          self.release(job)
        else:
          # unknown status, should not happen
          self.log(f"guard(), unknown job status: {job.status}", level=logging.CRITICAL)
          
        prev_job_status[job_id] = job.status
      
      if active_jobs != prev_active_jobs or self._record_time() - last_schedule_time > self.schedule_interval:
        # reset event occurred, either submit / release, or at least schedule_interval has passed        
        while self.schedule() is not None:
          pass
        prev_active_jobs = active_jobs

        last_schedule_time = self._record_time()

        # attempt to fast forward as scheduling events
        self.fast_forward()
        
        

      
      # check if executor is still alive
      for node_id in node_id_list:
        if node_id not in self.executor_server.connections:
          self.log(f"executor {node_id} disconnected", level=logging.ERROR)
          self.stop(wait_for_guard=False)
          
      time.sleep(guard_sleep_interval)

  """
  Utils
  """
  def _get_job_logger(self, job: Job) -> Logger:
    """
    Get the logger for the job.
    """
    job_logger = logging.getLogger(job.job_name)
    job_logger.setLevel(self.logger.level)
    for handler in self.logger.handlers:
      job_logger.addHandler(handler)
      
    return job_logger

  def _get_job_port(self) -> int:
    """
    Get the next available job port.
    """
    for port in range(self.scheduler_port_begin, 65535):
      if port not in self.job_port_usage:
        self.job_port_usage.add(port)
        return port
    raise RuntimeError(f"No available port for scheduler")
      
  def _free_job_port(self, port: int) -> None:
    """
    Free the job port.
    """
    # there appears to be resource leak. for experiment purpose, we never reuse a port as a temporary workaround
    return
    # if port in self.job_port_usage:
    #   self.job_port_usage.remove(port)
    # else:
    #   raise RuntimeError(f"Port {port} is not in use by scheduler")
  
  def _get_master_port(self, hostname: str) -> int:
    """
    Get the master port for the given hostname.
    """
    
    # find the next available port
    for port in range(self.worker_port_begin, 65535):
      if port not in self.worker_port_usage[hostname]:
      # if port not in self.worker_port_usage[hostname] \
      #   and port % 2 == 0: # only assign even port
        self.worker_port_usage[hostname].add(port)
        return port
    
    raise RuntimeError(f"No available port for {hostname}")
  
  def _free_master_port(self, hostname: str, port: int) -> None:
    """
    Free the master port for the given hostname.
    """
    # there appears to be resource leak. for experiment purpose, we never reuse a port as a temporary workaround
    return
    # if port % 2 == 1:
    #   port ^= 1
    # if port in self.worker_port_usage[hostname]:
    #   self.worker_port_usage[hostname].remove(port)
    # else:
    #   raise RuntimeError(f"Port {port} is not in use by {hostname}")

  def _get_active_jobs(self, locking: bool = False) -> List[Job]:
    """
    Get all active jobs 
    WAITING, PENDING, RUNNING but not finished (COMPLETED, FAILED, CANCELLED)
    Interface to policy
    """
    if locking:
      self.global_lock.acquire()
    
    active_jobs = [
      self.jobs[job_id] for job_id in self.active_jobs
    ]

    if locking:
      self.global_lock.release()
    
    return active_jobs
  
  def _get_waiting_jobs(self, locking: bool = False) -> List[Job]:
    """
    Get all waiting jobs
    Interface to policy
    """
    if locking:
      self.global_lock.acquire()
        
    waiting_jobs = [
      job for job in self._get_active_jobs(locking=False) if job.status == JobStatus.WAITING
    ]

    if locking:
      self.global_lock.release()
    
    return waiting_jobs
  

  def _get_running_jobs(self, locking: bool = False) -> List[Job]:
    """
    Get all running jobs
    Interface to policy
    """
    if locking:
      self.global_lock.acquire()
        
    running_jobs = [
      job for job in self._get_active_jobs(locking=False) if job.status == JobStatus.RUNNING
    ]

    if locking:
      self.global_lock.release()
    
    return running_jobs

  def _get_waiting_and_running_jobs(self, locking: bool = False) -> List[Job]:
    """
    Get all running and waiting jobs
    Interface to policy
    """
    if locking:
      self.global_lock.acquire()
    
    waiting_and_running_jobs = [
      job for job in self._get_active_jobs(locking=False) if job.status in [JobStatus.WAITING, JobStatus.RUNNING]
    ]

    if locking:
      self.global_lock.release()
    
    return waiting_and_running_jobs
  
  """
  Statistics Related
  """
  def _record_time(self) -> float:
    """
    Time in seconds since the scheduler started
    """
    if not self.is_running:
      # raise RuntimeError("Scheduler is not running")
      self.log('attempt to record time while scheduler is not running', level=logging.WARNING)
    return time.time() - self.time_base + self.ff_offset
    
  def _record_job_submit(self, job: Job):
    self.job_statistics[job] = {}
    
    self.job_statistics[job]["submit_time"] = self._record_time()
    
    fields = ["start_time", "end_time", "reconfig_init_time", "reconfig_start_time", "reconfig_stop_time", "n_acc", "plan_config"]
    for field in fields:
      self.job_statistics[job][field] = []
    self.job_statistics[job]["iter_time"] = {}
    
  def _record_job_start(self, job: Job):
    self.job_statistics[job]["start_time"].append(self._record_time())

  def _record_job_end(self, job: Job):
    self.job_statistics[job]["end_time"].append(self._record_time())
  
  # reconfig init: reconfig issued from scheduler
  # reconfig start: blocking migration start
  # reconfig stop: blocking migration stop
  def _record_job_reconfig_init(self, job: Job):
    self.job_statistics[job]["reconfig_init_time"].append(self._record_time())
    self._record_job_iter_time(job, -1.0)

  def _record_job_reconfig_start(self, job: Job):
    self.job_statistics[job]["reconfig_start_time"].append(self._record_time())
  
  def _record_job_reconfig_stop(self, job: Job):
    self.job_statistics[job]["reconfig_stop_time"].append(self._record_time())
    
  def _record_job_n_acc(self, job: Job, n_acc: int):
    self.job_statistics[job]["n_acc"].append(n_acc)

  def _record_job_plan_config(self, job: Job, plan_config: Union[None, PlanConfig]):
    n_acc = plan_config.world_size() if plan_config is not None else 0
    self._record_job_n_acc(job, n_acc)
    self.job_statistics[job]["plan_config"].append(plan_config)
  
  def _record_job_iter_time(self, job: Job, iter_time: float):
    reconfig_cnt = len(self.job_statistics[job]["reconfig_init_time"])
    self.job_statistics[job]["iter_time"][reconfig_cnt - 1] = iter_time
  
  def _get_job_age(self, job: Job):
    return self._record_time() - self.job_statistics[job]["submit_time"]
  
  def _get_job_up_time(self, job: Job):
    stats = self.job_statistics[job]
    
    if not len(stats["start_time"]):
      # self.log(f"attmpt to _get_job_up_time for a job that hasn't started", level=logging.WARNING)
      return 0
    
    up_cnt = min(len(stats["start_time"]), len(stats["end_time"]))
    prev_up_time = sum(
      stats["end_time"][i] - stats["start_time"][i] for i in range(up_cnt)
    )

    if job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
      return self._record_time() - stats["start_time"][-1] + prev_up_time
    else:
      return prev_up_time
  
  def _get_job_reconfig_cost(self, job: Job, exclude_first: bool=False):
    stats = self.job_statistics[job]
    
    reconfig_cnt = min(len(stats["reconfig_start_time"]), len(stats["reconfig_stop_time"]))
    
    return sum(
      stats["reconfig_stop_time"][i] - stats["reconfig_start_time"][i]
      for i in range(int(exclude_first), reconfig_cnt))
    
  def _get_job_reconfig_cnt(self, job: Job):
    stats = self.job_statistics[job]
    
    return min(len(stats["reconfig_start_time"]), len(stats["reconfig_stop_time"]))
  
  def _get_job_avg_reconfig_cost(self, job: Job):
    reconfig_cnt = self._get_job_reconfig_cnt(job)
    if reconfig_cnt == 0:
      return 0.0
    return self._get_job_reconfig_cost(job) / reconfig_cnt

  def _get_job_n_acc(self, job: Job) -> int:
    n_accs = self.job_statistics[job]["n_acc"]
    n_acc = n_accs[-1] if len(n_accs) else 0
    return n_acc

  def _get_job_iter_time(self, job: Job) -> float:
    iter_times = self.job_statistics[job]["iter_time"]
    key = max(iter_times.keys(), default=-1)
    if key == -1:
      return -1.0
    return iter_times[key]









class GeneralScheduler(Scheduler):
  """
    Scheduler to support baseline: Metis, Sia, Rubick
    
    The key difference is it supports checkpoint-resume for reconfiguration, and thus requires different API.
    
    For job submit to this scheduler, the Job.other_context["checkpoint_resume"] field must be set to a per-job unique directory on NFS.
  """
  def __init__(self, 
               config: Union[os.PathLike, Dict] = "~/.suika.json", 
               policy: Optional[str] = None, 
               ff_enable: bool = True,
               schedule_interval: int = default_schedule_interval,
               logger: Optional[Logger] = None,
               force_checkpoint_resume: bool = False,
               start_after_init: bool = True
               ):
    super().__init__(config=config, 
                     policy=policy, 
                     ff_enable=ff_enable, 
                     schedule_interval=schedule_interval, 
                     logger=logger, 
                     start_after_init=start_after_init)
    self.force_checkpoint_resume = force_checkpoint_resume
    
  def submit(self, job: Job) -> int:
    assert job.other_context.get("checkpoint_resume", None) is not None, \
      "job.other_context['checkpoint_resume'] must be set"
    return super().submit(job)
    
  def launch(self, job: Job, plan_config: Optional[PlanConfig], **kwargs) -> bool:
    if self._require_checkpoint_resumse(plan_config):
      kwargs["checkpoint_resume"] = job.other_context["checkpoint_resume"]
    return super().launch(job, plan_config, **kwargs)
    
  def _require_checkpoint_resumse(self, plan_config: Optional[PlanConfig]) -> bool:
    if self.force_checkpoint_resume:
      return True
    
    from Suika.cluster.policy import PolicyDP, PolicyMetis, PolicyRubick, PolicySia
    
    if isinstance(self.policy, (PolicyMetis, PolicyRubick, PolicySia)):
    # if isinstance(self.policy, (PolicyDP, PolicyMetis, PolicyRubick, PolicySia)):
      return True
    
    if plan_config is None:
      return True
    return False
  
  
  
  
"""
Simulator Scheduler
  A scheduler to simulate the scheduling process.
  It relies on the 'model_strategy' field in job.other_context to get estimator_C and n_iter
    addtionally, n_iter may be overridden by 'n_iter' field
"""
class SimulatorScheduler(GeneralScheduler):
  
  class Event(Enum):
    SUBMIT = 0 # context: job
    LAUNCH = 1 # context: (job, plan_config)
    COMPLETE = 2 # context: (job, version)
    SCHEDULE = 3 # context: None
    RECONFIG_BLOCK = 4 # context: (job)
    RECONFIG_COMPLETE = 5 # context: (job, reconfig_blocking_start_time, is_reconfig)
    # no support for cancel / fail
    
  event_list: List[Tuple[float, Event, Any]]
  cur_time: float
  last_schedule_time: float
  
  job_exec_status: Dict[Job, Tuple[float, int, float, int]] # last reconfig time, remaining n iter, throughput, version
  job_version: Dict[Job, int] # job version (number of reconfigs)
  
  launch_blocking_job: Set[Job] # jobs that are blocked by reconfiguration (if set not empty, scheduling should be blocked)
  last_schedule_unblock_time: int
  
  # scaling throughput decay function
  #   input: ratio of scaling up (#acc after scale up / #acc of user defined plan)
  #   ouput: ratio of throughput decay
  # default to 0.9 ** (log2(scale_ratio))
  # note that this is in additional to estimator to compensate for underutilization.
  scaling_throughput_decay: Callable[[float], float] 
  
  # reconfig cost function:
  #   input: n_acc, checkpoint_resumes
  #   output: pending time, blocking time in seconds
  # default to:
  #   clamp(0.5 * n_acc^2, 20, 80), clamp(0.5 * n_acc^2, 20, 80) if checkpoint resume
  #   clamp(0.5 * n_acc^2, 20, 80), clamp(n_acc, 10, 20)  if migration
  reconfig_cost: Callable[[int, bool], Tuple[float, float]]
  
  schedule_overhead: List[float] # record schedule overhead time in seconds
  
  def __init__(self, 
               config: Union[os.PathLike, Dict] = "~/.suika.json", 
               schedule_interval: int = default_simu_schedule_interval,
               force_checkpoint_resume: bool = False,
               policy: Optional[str] = None,
               add_tqdm: bool = True, 
               add_tqdm_refresh_at_event: bool = False,
               scaling_throughput_decay: Callable[[float], float] = None,
               reconfig_cost: Callable[[int, bool], Tuple[float, float]] = None,
               logger: Optional[Logger] = None):
    """
    Initialize the scheduler instance.
    """
    
    super().__init__(config=config, 
                     policy=policy, 
                     ff_enable=False, 
                     schedule_interval=schedule_interval, 
                     force_checkpoint_resume=force_checkpoint_resume, 
                     logger=logger, 
                     start_after_init=False)
    self.event_list = []
    self.cur_time = 0.0
    self.last_schedule_time = 0.0
    self.job_exec_status = {}
    self.job_version = {}
    self.progress_bar = add_tqdm
    self.progress_bar_refresh_at_event = add_tqdm_refresh_at_event
    self.launch_blocking_job = set()
    self.last_schedule_unblock_time = 0.0
    self.schedule_overhead = []
    
    self.is_running = True
    
    def default_scaling_throughput_decay(scale_ratio):
      if scale_ratio <= 1:
        return 1.0
      else:
        return 0.94 ** (math.log2(scale_ratio))
    
    if scaling_throughput_decay is not None:
      self.scaling_throughput_decay = scaling_throughput_decay
    else:
      self.scaling_throughput_decay = default_scaling_throughput_decay
      
    def default_reconfig_cost(n_acc: int, checkpoint_resume: bool) -> float:
      clamp = lambda x, min_v, max_v: max(min(x, max_v), min_v)
      if checkpoint_resume:
        return clamp(0.5 * n_acc ** 1.5, 10, 120), clamp(0.5 * n_acc ** 1.5, 10, 120)
      else:
        return clamp(0.5 * n_acc ** 1.5, 10, 30), clamp(0.5 * n_acc, 2, 30)
      
    if reconfig_cost is not None:
      self.reconfig_cost = reconfig_cost
    else:
      self.reconfig_cost = default_reconfig_cost
    
    
  """
    submit a job before simulate.
  """
  def submit(self, job: Job, time: float) -> None:
    self._add_event(time, event=self.Event.SUBMIT, context=job)

  """
    start the simulation.
  """
  def simulate(self) -> None:
    if self.progress_bar:
      from tqdm import tqdm
      self.progress_bar = tqdm(total=len(self.event_list), desc=f"simulation-{get_policy_name(self.policy)}", unit="job")
    
    if not len(self.event_list):
      raise RuntimeError("No event to simulate; call submit() before simulate()")

    while self._simulate_step():
      pass
    
    if self.progress_bar:
      self.progress_bar.close()

  def _simulate_step(self) -> bool:
    import time as time_module
    
    if not len(self.event_list):
      return False # simulation completed
        
    # get the next event
    time, event, context = self.event_list.pop(0)
    
    # check if a periodic schedule is needed, if so, revert the event and add a schedule
    if not self._schedule_blocked() and self.last_schedule_unblock_time < self.last_schedule_time + self.schedule_interval < time:
      self._add_schedule(self.last_schedule_time + self.schedule_interval)
      self._add_event(time, event=event, context=context)
      return True

    # advance cur_time
    if time < self.cur_time - 1e-2: 
      # note the eps here is 1e-2, other padding should use 1e-3 or smaller
      raise RuntimeError("Simulation time is not monotonic; check implementation bug")
    self.cur_time = time
    if self.progress_bar:
      datetime = time_module.strftime("%H:%M:%S", time_module.localtime(time_module.time()))
      util = f"{self.rm.get_utilization() * 100:.1f}%"
      self.progress_bar.set_description(
        f"simulation-{get_policy_name(self.policy)}@[v.{self.cur_time / 3600:.2f}h|r.{datetime}|u.{util}]",
        refresh=self.progress_bar_refresh_at_event
      )
    
    # process the event
    if event == self.Event.SUBMIT:
      job = context
      self._submit(job)
    elif event == self.Event.SCHEDULE:
      schedule_start_time = time_module.time()
      
      while not self._schedule_blocked():
        if self._schedule():
          pass
        else:
          break
        
      schedule_end_time = time_module.time()
      self.schedule_overhead.append(schedule_end_time - schedule_start_time)
      
    elif event == self.Event.COMPLETE:
      job, version = context
      if version < self.job_version[job]:
        pass # job has been reconfigured, this complete event is outdated
      else:
        self._release(job)
        if self.progress_bar is not None:
          self.progress_bar.update(1)
          
    elif event == self.Event.RECONFIG_BLOCK:
      job = context
      self._reconfig_block_start(job)
    elif event == self.Event.RECONFIG_COMPLETE:
      job, reconfig_start_time, is_reconfig = context
      self._reconfig_complete(job, reconfig_start_time, is_reconfig)
    else:
      raise RuntimeError(f"Unknown event {event} in simulation")
              
    return True
  
  def _submit(self, job: Job):
    job.job_id = self.job_id_counter
    
    self.job_id_counter += 1
    job.status = JobStatus.WAITING
    self.policy.submit(job)
    
    self.jobs[job.job_id] = job
    self.active_jobs.add(job.job_id)
    
    estimator_C = job.other_context["model_strategy"].model_config.estimator_C
    job.job_optimizer = JobOptimizer(rm=self.rm, C=estimator_C)
    # job.job_optimizer = JobOptimizer(rm=self.rm, C=estimator_C, logger=self.logger)

    self._add_event(self.cur_time, event=self.Event.SCHEDULE, context=None)
    
    self._record_job_submit(job)
    self.log(f"submit(), {job} at {self.cur_time}")
    
  def _schedule(self):    
    self.last_schedule_time = self.cur_time

    decision = self.policy.schedule()
    if decision is not None:
      job, plan_config = decision
      self.log(f"schedule() selected, {job} at {self.cur_time:.1f}")
      if self._launch(job, plan_config):
        self.policy.launch_success_callback(job, plan_config)
        return True
      else:
        self.policy.launch_failure_callback(job, plan_config)
    else:
      self.log(f"schedule() idled at {self.cur_time}")
      return False
  
  def _launch(self, job: Job, plan_config: PlanConfig) -> bool:
    cur_acc_mesh = job.cur_plan_config.acc_mesh if job.cur_plan_config else []
    new_acc_mesh = plan_config.acc_mesh if plan_config is not None else []
    cur_n_acc = 0 if job.cur_plan_config is None else job.cur_plan_config.world_size()
    new_n_acc = 0 if plan_config is None else plan_config.world_size()
    
    if not job.status in [JobStatus.WAITING, JobStatus.RUNNING]:
      if job.status == JobStatus.PENDING:
        raise RuntimeError(f"Cannot launch job {job} in PENDING status (as this should not happen in simulation, we raise error)")
      return False
    if not self.rm.raw_free(cur_acc_mesh, strict=True):
      raise RuntimeError("Failed to free resources; external error, check your policy impl")
    if not self.rm.raw_allocate(new_acc_mesh, strict=True):
      raise RuntimeError("Failed to allocate resources; external error, check your policy impl")
    
    # if reconfig
    is_reconfig = job.status == JobStatus.RUNNING
    is_shrink = new_n_acc <= cur_n_acc or (self._require_checkpoint_resumse(plan_config) and new_n_acc == cur_n_acc)
    
    self._record_job_plan_config(job, plan_config)
    self._record_job_reconfig_init(job)
    
    if not is_reconfig:
      self._record_job_start(job)
    if is_shrink:
      self.launch_blocking_job.add(job)
    
    # update status
    job.status = JobStatus.PENDING
    
    # record job reconfig plan config
    if plan_config is not None:
      plan_config.set_rank_mesh()
    job.cur_plan_config = plan_config      
    

    # add reconfig event 
    pending_time, blocking_time = self.reconfig_cost(max(new_n_acc, cur_n_acc), self._require_checkpoint_resumse(plan_config))
    blocking_time = max(min(blocking_time, pending_time), 1e-3)
    
    reconfig_blocking_start_time = self.cur_time + pending_time - blocking_time
    reconfig_complete_time = self.cur_time + pending_time
    self._add_event(reconfig_blocking_start_time, event=self.Event.RECONFIG_BLOCK, context=job)
    self._add_event(reconfig_complete_time, event=self.Event.RECONFIG_COMPLETE, context=(job, reconfig_blocking_start_time, is_reconfig))
    
    self.log(f"launch() initiated: {job} at {self.cur_time:.1f}, n_acc={new_n_acc}", level=logging.DEBUG)
    return True

  def _reconfig_block_start(self, job: Job):
    # update version
    self._record_job_reconfig_start(job)
    version = self.job_version.get(job, 0) + 1
    self.job_version[job] = version
    
  def _reconfig_complete(self, job: Job, reconfig_blocking_start_time: float, is_reconfig: bool) -> None:
    if job.status != JobStatus.PENDING:
      assert job.status == JobStatus.COMPLETED, f"reconfig_complete() failed, {job} is not in PENDING state nor has it completed"
      return
    if job in self.launch_blocking_job:
      # add a schedule, or the simulation will terminate prematurely
      self._add_schedule(self.cur_time)
      self._remove_launch_blocking_job(job)
      
    plan_config = job.cur_plan_config
    version = self.job_version[job]
    self._record_job_reconfig_stop(job)
    
    
    if plan_config is None:
      # preempt
      job.status = JobStatus.WAITING
      
      remaining_n_iter = self._get_remaining_n_iter(job, reconfig_blocking_start_time)
      self.job_exec_status[job] = (self.cur_time, remaining_n_iter, 0.0, version)
      
      self.policy.submit(job)
      self._record_job_end(job)
      self.log(f"preempt(), {job} at {self.cur_time}")
      return
    
    # launch / reconfig

    # update job status and exec status
    job.status = JobStatus.RUNNING
    
    new_n_acc = plan_config.world_size()
    usr_n_acc = job.usr_plan_config.world_size()
    comp_decay = self.scaling_throughput_decay(new_n_acc / usr_n_acc)
    # throughput = job.job_optimizer.eval_throughput(plan_config, log_prefix=f"{job.job_name} at {self.cur_time:.1f}", comp_decay=comp_decay)
    throughput = job.job_optimizer.eval_throughput(plan_config, log_prefix=f"{job.job_name} at {self.cur_time:.1f}", comp_decay=comp_decay)
    # throughput *= comp_decay
    self._record_job_iter_time(job, 1 / throughput)
    
    remaining_n_iter = self._get_remaining_n_iter(job, reconfig_blocking_start_time)
    self.job_exec_status[job] = (self.cur_time, remaining_n_iter, throughput, version)
    
    if throughput == 0:
      raise RuntimeError(f"job infeasible for {job.job_name} at time {self.cur_time:.1f}: {plan_config}")
    
    # add complete event
    predicted_completion = self.cur_time + remaining_n_iter / throughput
    self._add_event(predicted_completion, event=self.Event.COMPLETE, context=(job, version))
        
    self.log(f"launch(), {job}@{'reconfig' if is_reconfig else 'launch'} at {self.cur_time:.1f}")
    self.log(f"launch(), {job}@{'reconfig' if is_reconfig else 'launch'} at {self.cur_time:.1f}\nthroughput={throughput:.3f}, latency={1/throughput:.3f}, predicted_completion: {predicted_completion}, n_acc={new_n_acc}, pc={plan_config}", level=logging.DEBUG)

  def _release(self, job: Job) -> None:
    # release the job
    job.status = JobStatus.COMPLETED
    
    # free resources
    self.rm.raw_free(job.cur_plan_config.acc_mesh, strict=True)
    
    # remove from active jobs
    self.active_jobs.remove(job.job_id)
    # record statistics
    self._record_job_end(job)
    
    # remove launch blocking
    self._remove_launch_blocking_job(job)
        
    self._add_schedule(self.cur_time)
    
    self.log(f"release(), {job} at {self.cur_time}")
    
  def _add_schedule(self, time: float):
    self._add_event(time, event=self.Event.SCHEDULE, context=None)  
  
  def _add_event(self, time: float, event: Event, context: Any):
    self.event_list.append(
      (time, event, context)
    )
    self.event_list.sort(key=lambda x: x[0])
  
  def _get_remaining_n_iter(self, job: Job, time: float) -> float:
    if job not in self.job_exec_status:
      if "n_iter" in job.other_context:
        return float(job.other_context["n_iter"])
      else:
        return float(job.other_context["model_strategy"].n_iter)
    
    last_reconfig_time, remaining_n_iter, throughput, _ = self.job_exec_status[job]
    
    if (time - last_reconfig_time) * throughput > remaining_n_iter:
      raise RuntimeError("Job completed but not detected")
    
    # do not round down to avoid fractional loss... keep float
    return remaining_n_iter - (time - last_reconfig_time) * throughput
    
  def _record_time(self) -> float:
    return self.cur_time
  
  def _schedule_blocked(self) -> bool:
    return len(self.launch_blocking_job) > 0
  
  def _remove_launch_blocking_job(self, job):
    if job in self.launch_blocking_job:
      self.launch_blocking_job.remove(job)
      if len(self.launch_blocking_job) == 0:
        self.last_schedule_unblock_time = self._record_time()
  
"""
@EXPORT
"""
def analyze_job_statistics(job_statistics: Union[Dict[Job, Dict[str, float]], Scheduler]) -> Dict[str, Any]:
  """
  Analyze the job statistics.
  
  Returns:
    dict of {
      avg_jct, 
      avg_w_jct,
      avg_duration, 
      avg_queuing_delay, 
      avg_gpu_time,
      make_span, 
      per_job_avg_reconfig_cnt,
      per_job_avg_reconfig_cost, 
      per_reconfig_avg_reconfig_cost,
      job_status (count of each job status),
      per_job_statistics (dict of job -> dict of statistics)
    }
  """
  if isinstance(job_statistics, Scheduler):
    job_statistics = job_statistics.job_statistics
  
  avg_jct = 0.0
  avg_w_jct = 0.0
  avg_duration = 0.0
  avg_queuing_delay = 0.0
  make_span = 0.0
  avg_gpu_time = 0.0
  
  total_usr_n_acc = 0
  
  total_reconfig_cost = 0.0
  reconfig_count = 0
  
  per_job_avg_reconfig_cnt = 0.0
  per_job_avg_reconfig_cost = 0.0
  per_reconfig_avg_reconfig_cost = 0.0
  
  job_status = {
    k: 0 for k in JobStatus
  }
  
  per_job_statistics = {}
  
  for job, stats in job_statistics.items():
    try:
      jct = stats["end_time"][-1] - stats["submit_time"]
    except IndexError:
      jct = float("inf")
    try:
      duration = stats["end_time"][-1] - stats["start_time"][0]
    except IndexError:
      duration = float("inf")
    usr_n_acc = job.usr_plan_config.world_size()
    w_jct = usr_n_acc * jct    
    
    try:
      queuing_delay = stats["start_time"][0] - stats["submit_time"]
    except IndexError:
      queuing_delay = float("inf")
    try:
      reconfig_time = stats["reconfig_start_time"] + [stats["end_time"][-1]]
      while len(reconfig_time) < len(stats["n_acc"]) + 1:
        stats["n_acc"].pop(-1) # possible, if last reconfig overlap with completion, then reconfig start may not be recorded (and is not valid either)
      gpu_time = 0
      for i in range(len(stats["n_acc"])):
         gpu_time += stats["n_acc"][i] * (reconfig_time[i + 1] - reconfig_time[i])
    except IndexError as e:
      print(stats)
      raise e
      
    avg_jct += jct
    avg_w_jct += w_jct
    avg_duration += duration
    avg_queuing_delay += queuing_delay
    avg_gpu_time += gpu_time
    make_span = max(make_span, max(stats["end_time"] + [0.0]))
    
    reconfig_cost = [
      stats["reconfig_stop_time"][i] - stats["reconfig_start_time"][i]
      for i in range(len(stats["reconfig_stop_time"]))
    ]
    
    #? if job is complete preempted, launch other than first launch is also counted at reconfig cost.
    total_reconfig_cost += sum(reconfig_cost[1:]) # exclude first launch
    reconfig_count += len(reconfig_cost[1:])
    total_usr_n_acc += usr_n_acc
    
    job_status[job.status] += 1
    
    per_job_statistics[job] = {
      "jct": jct,
      "w_jct": w_jct,
      "duration": duration,
      "queuing_delay": queuing_delay,
      "gpu_time": gpu_time,
    }
    
  num_jobs = len(job_statistics)
  if num_jobs > 0:
    avg_jct /= num_jobs
    avg_queuing_delay /= num_jobs
    avg_duration /= num_jobs
    per_job_avg_reconfig_cnt = reconfig_count / num_jobs
    per_job_avg_reconfig_cost = total_reconfig_cost / num_jobs
    avg_gpu_time /= num_jobs
  if total_usr_n_acc > 0:
    avg_w_jct /= total_usr_n_acc
  
  if reconfig_count > 0:
    per_reconfig_avg_reconfig_cost = total_reconfig_cost / reconfig_count
  else:
    per_reconfig_avg_reconfig_cost = 0.0
    
  job_status = {
    k.name: v for k, v in job_status.items() if v > 0
  }
    
  return {
    "avg_jct": avg_jct,
    "avg_w_jct": avg_w_jct,
    "avg_duration": avg_duration,
    "avg_queuing_delay": avg_queuing_delay,
    "avg_gpu_time": avg_gpu_time,
    "make_span": make_span,
    "per_job_avg_reconfig_cnt": per_job_avg_reconfig_cnt,
    "per_job_avg_reconfig_cost": per_job_avg_reconfig_cost,
    "per_reconfig_avg_reconfig_cost": per_reconfig_avg_reconfig_cost,
    "job_status": job_status,
    "per_job_statistics": per_job_statistics
  }