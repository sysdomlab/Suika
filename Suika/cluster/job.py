import queue
import shlex
from enum import Enum
import logging
from logging import Logger
import time
from typing import Callable, List, Optional, Dict, Any, Tuple, Union
import threading
import os


from Suika.cluster.estimator import JobOptimizer, AcceleratorMesh1D, AcceleratorMesh2D, AcceleratorMesh3D
from Suika.cluster.commu import InterProcessFlag, Job2WorkerMsg, SuikaSocketClient, SuikaSocketServer, Worker2JobMsg, Executor2SchedulerMsg, Scheduler2ExecutorMsg
from Suika.cluster.plan_config import PlanConfig
from Suika.cluster.resource import Accelerator, AcceleratorMesh, HostnameMesh, ResourceManager
from Suika.device_mesh import RawDeviceMesh3D
from Suika.utils import kill_all_child_processes, kill_processes, SUIKA_LOG_DIR

"""
Job Status
"""
class JobStatus(Enum):
  """
  UNKNOWN -> WAITING -> PENDING -> RUNNING -> COMPLETED/FAILED/CANCELLED
                        A          |
                        |──────────|
                        (reconfig)
  """
  WAITING = 0
  PENDING = 1
  RUNNING = 2
  COMPLETED = 3
  FAILED = 4
  CANCELLED = 5
  UNKNOWN = 6
  
"""
Worker Status
"""  
class WorkerStatus(Enum):
  PENDING = 0
  RUNNING = 1
  COMPLETED = 2
  FAILED = 3
  SHRANK = 4

"""
Necessary Config for Launching a Job
"""
class LaunchConfig():
  master_addr: str
  master_port: str
  
  world_size: int
  
  rank_mesh: RawDeviceMesh3D
  acc_mesh: AcceleratorMesh
  hostname_mesh: HostnameMesh
  
  # communication endpoint for worker process to job manager 
  # in worker's view, job manager is the scheduler agent
  job_endpoint: str

  
  def __init__(self):
    self.master_addr = None
    self.master_port = None
    
    self.world_size = None
    
    self.rank_mesh = None
    self.acc_mesh = None
    self.hostname_mesh = None
    
    self.job_endpoint = None

"""
Job instance in cluster view

Also the job manager for this job if launched
"""
class Job():
  # basic info
  job_id: Optional[int] # assigned by scheduler at submit
  job_name: Optional[str] # spec by user
  
  cwd: str # spec by user
  cmd: str # spec by user
  usr_plan_config: PlanConfig # spec by user at initial submit, a virtual config

  status: JobStatus
  
  # launch config
  launch_config: Optional[LaunchConfig] # assigned by scheduler at launch
  
  # socket
  job_server: SuikaSocketServer # connect to worker # created at launch

  # runtime
  job_optimizer: JobOptimizer # created at launch

  cur_plan_config: PlanConfig # calcuated by Job Level Optimizer from launch config at launch / reconfig, a physical config
  
  profile: Dict[str, Any] # profiling data from worker
  guard_thread: Optional[threading.Thread] # created at launch
  guard_lock: threading.Lock # lock for guard thread
  
  worker_status: Dict[int, WorkerStatus] # active ranks in the job, used to track if all workers are alive (launch failure / complete are False, but runtime failure is not tracked)
  
  reconfig_start_timers: List[Callable] # list of functions to call when reconfig starts for precise timing
  reconfig_stop_timers: List[Callable] # list of functions to call when reconfig stop for precise timing
  
  # logger
  logger: Optional[Logger] # assigned at launch, same as the scheduler's

  # othter context
  other_context: Dict[str, Any] # other context for job, such as model config, etc.
  
  # ff info
  record_iter_time: Callable[[float], None] # record iteration time for job statisitcs
  scheduler_timer: Callable[[], float] # get virtual scheduler time, set at submit # deprecated
  ff_infos: List[Tuple[Tuple[float, float], int, float]] # list of history (now(wallclock), now(virtual), cur_iter, iter_time) tuple
    
  # create a job from user's view to submit
  def __init__(self, job_name: str, cmd: str, plan_config: PlanConfig, cwd = "./", other_context: Optional[Dict[str, Any]] = None):
    self.job_id = None
    self.job_name = job_name
    
    self.cwd = cwd
    self.cmd = cmd
    self.usr_plan_config = plan_config
    
    self.status = JobStatus.UNKNOWN 
        
    self.launch_config = None
    
    self.job_server = None

    self.job_optimizer = None

    self.cur_plan_config = None
    self.guard_thread = None
    self.guard_lock = None
    self.reconfig_thread = None
    
    self.worker_status = {}
    
    self.reconfig_start_timers = []
    self.reconfig_stop_timers = []

    self.logger = None
    
    self.other_context = other_context if other_context is not None else {}
    
    self.record_iter_time = lambda x: None
    self.scheduler_timer = lambda x: 0.0
    self.ff_infos = []


  """
  Scheduler Interface
  """
  
  """
  Launch the job with given LaunchConfig
  """
  def launch(self, 
             plan_config: PlanConfig, 
             launch_config: LaunchConfig, 
             executor_server: SuikaSocketServer,
             rm: ResourceManager, 
             reconfig_start_timer: Optional[Callable] = None,
             reconfig_stop_timer: Optional[Callable] = None,
             checkpoint_resume: Optional[str] = None, 
             logger: Optional[Logger] = None) -> bool:
    self.log("launch start")
    
    # check job status
    if self.status != JobStatus.WAITING:
      self.log(f"launch: incorrect status, expect WAITING, got {self.status}", level=logging.CRITICAL)
      return False
    
    # save args
    # launch_config is saved in reconfig
    self.logger = logger

    # initialize socket
    self.job_server = SuikaSocketServer(launch_config.job_endpoint, 
                                        # logger=logger
                                        )
    self.job_server.start() # start right now, as reconfig requires

    # initialize job optimizer
    self.job_optimizer = JobOptimizer(
      rm = rm,
      logger = logger
    )
    
    # initialize guard
    self.guard_lock = threading.Lock()
    self.guard_thread = threading.Thread(target = self.guard)
    
    # launch proc via reconfig (unified interface)
    if not self.reconfig(
      plan_config=plan_config, 
      launch_config=launch_config,
      executor_server=executor_server,
      reconfig_start_timer=reconfig_start_timer,
      reconfig_stop_timer=reconfig_stop_timer,
      checkpoint_resume=checkpoint_resume,
      ):
      return False
    
    # start guard (must after self.reconfig, otherwise earily exit due to incorrect self.status)
    self.log("job launch: guard process")
    self.guard_thread.start()
    
    self.log("job launch: complete")
    return True
  
  """
  @name: reconfig
  @brief: Accept a new plan_config and launch_config,
          launch necessary new workers
          and send configuration to all workers.
          
          plan_config and launch_config are assumed to be consistent across reconfiguration.
          
          this interface is consistent for both launch and reconfiguration
          * while its a bit redundant... policy request eval speed-up plan_config from job,
            then decide the best job, and send plan_config back to job
          * interface is kept uniform
          
  @param plan_config: new plan config, if None, preempt the job
  """
  
  def reconfig(self, 
               plan_config: PlanConfig, 
               launch_config: LaunchConfig,
               executor_server: SuikaSocketServer,
               checkpoint_resume: Optional[str] = None, # checkpoint dir, if set, traditional checkpoint resume reconfig is used instead of migration
               reconfig_start_timer: Optional[Callable] = None, # for precise timing of reconfig overhead, passed in by scheduler
               reconfig_stop_timer: Optional[Callable] = None 
               ) -> bool:
    
    # threads to parallelize launching / sending reconfig
    threads: List[threading.Thread] = []
    timeout = 600 # in sec # general timeout for launching job

    # basic info for reconfig
    is_checkpoint_resume = checkpoint_resume is not None
    is_preempt = plan_config is None
    
    old_ws = getattr(self.launch_config, "world_size", 0)
    new_ws = getattr(launch_config, "world_size", 0)
    # case 1. for standard migration, they must be of same set of GPUs, and blocking is in fact not needed
    # csae 2. for checkpoint resume, the sets of GPUs may switch, and must block launching before return
    is_shrink = new_ws < old_ws or (is_checkpoint_resume and new_ws == old_ws)

    # some helper function
    def threads_start_and_join():
      nonlocal threads
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()
      threads = []

    def get_env(acc: Accelerator, rank: int):
      # though LOCAL_RANK is provided, user program should use cuda:0 as device, CUDA_VISIBLE_DEVICES is set to only one accelerator
      env = {
        "MASTER_ADDR": str(launch_config.master_addr),
        "MASTER_PORT": str(launch_config.master_port),
        "RANK": str(rank),
        # "LOCAL_RANK": str(acc.local_id),
        "WORLD_SIZE": str(launch_config.world_size),
        "JOB_ENDPOINT": str(launch_config.job_endpoint),
        "CUDA_VISIBLE_DEVICES": str(acc.local_id),
        "JOB_NAME": str(self.job_name), # for debugging
        "SUIKA_LOG_DIR": SUIKA_LOG_DIR,
      }
      return env
    
    def send_reconfig_func(acc: Accelerator, rank: int, hostname: str):
        
      def exec(ctx = (acc, rank, hostname)):
        acc, rank, hostname = ctx
        if (is_checkpoint_resume or rank >= old_ws):
          # problem: guard is blocked, so worker_status not updated, but old process may have completed in this short window. (though never happened in my experiment so far)
          # only wait for newly launched processes
          # if old process happens to complete during reconfiguration, as we do not wait for it, it won't hang or fail.
          # another safeguard here is that, after sending complete to scheduler, worker side client wait for ack before exiting, so the connection is still alive for a while.
          # still, the code below incorrectly set worker status to pending, but when guard is unblocked, completion message will be received.
          if not self.job_server.wait_for(rank, timeout = timeout):
            self.log(f"({rank}) does not connect within time, job failed")
            self._handle_failed(rank)
            return
        
        real_hostname = self.job_server.connections[rank].getpeername() if rank in self.job_server.connections else None
        self.log(f"({rank}) send reconfig to {hostname} / {real_hostname}")

        if self.worker_status[rank] != WorkerStatus.RUNNING or real_hostname is None:
          self.log(f"({rank}) worker not in RUNNING ({self.worker_status[rank]}) or connection is lost ({real_hostname}), skipping sending reconfig", level=logging.WARNING)
          return
        self.worker_status[rank] = WorkerStatus.PENDING
        
        retry = 3
        while retry > 0:
          retry -= 1
          # send checkpoint_resume (possibily None)
          self.job_server.send(
            dst=rank,
            obj=(Job2WorkerMsg.CHECKPOINT_RESUME, checkpoint_resume)
          )

          # send new plan_config and env
          env = get_env(acc, rank)
          self.job_server.send(
            dst=rank,
            obj=(Job2WorkerMsg.RECONFIG, (plan_config, env))
          )

          time.sleep(1) # resend three times (not sure why package get lost sometime)
      
      thread = threading.Thread(target=exec)
      threads.append(thread)
      
    
    def launch_func(acc: Accelerator, rank: int, hostname: str):
      # if rank in self.worker_status, skip launching them anew:   
      if rank in self.worker_status:
        # do not check for shrank, shrank worker's status entry is harvested by guard.
        return
      
      env = get_env(acc, rank)
      env_str = " ".join([f"{k}={v}" for k, v in env.items()])
      
      log_dir = shlex.quote(f"{SUIKA_LOG_DIR}/{self.job_name}")
      log_file = f"{log_dir}/main_[{rank}].log"
      quoted_cmd = shlex.quote(f"cd {self.cwd} && (mkdir -p {log_dir} || true) && export {env_str} && {self.cmd} > {log_file} 2>&1")
      # the OR true avoids occasional NFS glitch
      
      exec_cmd = f"bash -c {quoted_cmd}"

      self.log(f"command: {exec_cmd}")
      
      def exec(ctx = (exec_cmd, rank)):
        exec_cmd, rank = ctx
        # self.log(f"({rank}) launching job on {hostname}")
        
        self.worker_status[rank] = WorkerStatus.PENDING
        
        # send exec command to executor server
        executor_server.send(
          dst=acc.node_id,
          obj=(Scheduler2ExecutorMsg.EXEC, (rank, exec_cmd))
        )
        
        # wait for feedback
        feedback = executor_server.recv(rank=acc.node_id, wait=True, timeout=timeout)        
        
        if feedback is not None: # note that we may receive some feedback from other rank on the same node. but the total recv number is matched. and we set to pending prior to send message
          msg_type, recv_rank = feedback
          if msg_type == Executor2SchedulerMsg.SUCCEEDED:
            # set new worker's to RUNNING temporarily, as it's waiting for first reconfig to start (which has not been issued till send_reconfig_func)
            self.worker_status[recv_rank] = WorkerStatus.RUNNING
          elif msg_type == Executor2SchedulerMsg.FAILED:
            self.worker_status[recv_rank] = WorkerStatus.FAILED
        else:
          self.log(f"timeout waiting for executor feedback.")
          # unsure if this rank is recv by other exec, or just timeout. leave as pending. one failure means all failure anyway
      thread = threading.Thread(target=exec)
      threads.append(thread)

    
    """
    ===========================
    beginning of reconfig logic
    ===========================
    """
  
    self.log("job reconfig start")
    # check status before set to PENDING
    if self.status == JobStatus.PENDING:
      self.log("job reconfig: last reconfig still PENDING, rejecting", level=logging.WARNING)
      return False
          
    if self.status not in [JobStatus.WAITING, JobStatus.RUNNING]:
      self.log(f"job reconfig: incorrect status, expect WAITING or RUNNING, got {self.status}", level=logging.CRITICAL)
      return False
    
    if is_preempt and (self.status == JobStatus.WAITING or not is_checkpoint_resume):
      self.log(f"job reconfig: plan_config is None, but job is either WAITING (status={self.status}) or not using checkpoint-resume (value={is_checkpoint_resume}); this should not happen, check your code", level=logging.CRITICAL)
      return False
    
    # confirmed to be valid reconfig at this point

    # acquire guard lock to avoid Job.status mutation
    self.guard_lock.acquire()
    if any(worker_status != WorkerStatus.RUNNING for worker_status in self.worker_status.values()):
      self.log(f"job reconfig: some worker not in RUNNING status ({self.worker_status}), rejecting", level=logging.WARNING)
      self.guard_lock.release()
      return False
    
    # set status to PENDING, block profile update
    self.status = JobStatus.PENDING
    self.job_optimizer.est.set_block_update_constants(True)
    
    if reconfig_start_timer:
      if not is_checkpoint_resume:
        self.reconfig_start_timers.append(reconfig_start_timer)        
      else:
        reconfig_start_timer() # prior timing method is based on RECONFIG_START message from worker 0, which is not properly sent during checkpoint-resume.
    if reconfig_stop_timer:
      self.reconfig_stop_timers.append(reconfig_stop_timer)
    
    self.log("job reconfig: launch new workers (if any)")
    
    # save launch config and plan_config. set rank_mesh.
    prev_lc = self.launch_config
    if not is_preempt:
      self.launch_config = launch_config
      plan_config.rank_mesh = launch_config.rank_mesh
      self.cur_plan_config = plan_config
    else:
      launch_config = prev_lc # since get_env relies on this.
      self.launch_config = None
      self.cur_plan_config = None
    
    if not is_checkpoint_resume:
      # standard suike reconfig
      # send config to union of two plans' workers (i.e., the workers of the larger plan)
      if is_shrink:
        lc_to_send_reconfig = prev_lc
      else:
        lc_to_send_reconfig = launch_config
    else:
      # checkpoint-resume
      # always send to new plan's workers
      lc_to_send_reconfig = launch_config
      if prev_lc is not None:
        Job._recursive_launch(
          launch_func=send_reconfig_func,
          rank_mesh=prev_lc.rank_mesh,
          acc_mesh=prev_lc.acc_mesh,
          hostname_mesh=prev_lc.hostname_mesh
        )
        threads_start_and_join()

        # release the lock and wait for JobStatus change
        self.guard_lock.release()
        wait_for_checkpoint_interval = 0.1
        wait_for_checkpoint_timeout = 600
        wait_for_checkpoint_start = time.time()
        while self.status == JobStatus.PENDING and time.time() - wait_for_checkpoint_start < wait_for_checkpoint_timeout:
          time.sleep(wait_for_checkpoint_interval)
        
        if self.status == JobStatus.PENDING:
          self.log(f"job reconfig: job timeout during checkpointing", level=logging.CRITICAL)
          self._handle_failed()
          return False
        if self.status != JobStatus.WAITING:
          self.log(f"job reconfig: job finished during checkpointing ({self.status})", level=logging.WARNING)
          return False
        else:
          # note that we trickly reuse WorkerStatus.SRANK and JobStatus.PENDING -> WAITING change to monitor completion of checkpoint.
          # this may expose fake WAITING state of job for a short period.
          # but as scheduler side does not rely on such state change mechanism, and Job.reconfig holds global lock, it's fine.
          
          # at this moment, all workers status should have been cleared by guard, and guard itself quitted
          
          if not is_preempt:
            # not preemption
            self.guard_lock.acquire()
            # revert job status to PENDING (from WAITING)
            self.status = JobStatus.PENDING
            self.job_optimizer.est.set_block_update_constants(True)
            self.job_server.stop()
            # restart guard thread 
            self.guard_thread = threading.Thread(target=self.guard)
            self.guard_thread.start()
          else:
            # preemption
            self.guard_thread.join()
            self.status = JobStatus.WAITING
            self.log("job reconfig: preempted")
            return True
      
  
    # launch new workers
    Job._recursive_launch(
      launch_func=launch_func,
      rank_mesh=self.launch_config.rank_mesh, 
      acc_mesh=self.launch_config.acc_mesh, 
      hostname_mesh=self.launch_config.hostname_mesh
    )
    
    threads_start_and_join()
    
    failed_rank = []
    for rank, worker_status in self.worker_status.items():
      if worker_status != WorkerStatus.RUNNING:
        failed_rank.append(rank)
        self._handle_failed(rank)
    
    if failed_rank:
      self.log(f"launch failed on {failed_rank}")
      self.guard_lock.release()
      return False

    # detach to avoid blocking the scheduler
    def send_reconfig():
      # send config
      self.log("job reconfig: send reconfig")
      # to support shrink, we need to send reconfig to all workers even if some of them are to be removed.
      Job._recursive_launch(
        launch_func=send_reconfig_func,
        rank_mesh=lc_to_send_reconfig.rank_mesh, 
        acc_mesh=lc_to_send_reconfig.acc_mesh, 
        hostname_mesh=lc_to_send_reconfig.hostname_mesh
      )
      threads_start_and_join()

      # JobStatus.RUNNING is set by guard upon all worker reconfig complete
      # release guard lock
      self.guard_lock.release()

      
    # this thread is joined in guard.
    self.reconfig_thread = threading.Thread(target=send_reconfig)
    self.reconfig_thread.start()

    if is_shrink: 
      # blocking if shrinking
      # ensure reconfiguration complete for shrinking opeartion, otherwise following launch may fail due to OOM
      wait_for_shrink_start = time.time()
      wait_for_shrink_timeout = 600
      wait_for_shrink_interval = 0.1
      while self.status == JobStatus.PENDING and time.time() - wait_for_shrink_start < wait_for_shrink_timeout:
        time.sleep(wait_for_shrink_interval)
      
      if self.status == JobStatus.PENDING:
        self.log(f"job reconfig: job timeout during shrinking", level=logging.CRITICAL)
        self._handle_failed()
        return False
      if self.status not in [JobStatus.RUNNING, JobStatus.WAITING]:
        self.log(f"job reconfig: job finished during shrinking ({self.status})", level=logging.WARNING)
        return False

    self.log("job reconfig: complete")
    
    return True
  
  
  """
  eval_speed_up
  """
  def eval_speed_up(self, 
                    new_resources: AcceleratorMesh2D, 
  )-> Tuple[PlanConfig, float]: # percentage of speed with given acc
    return self.job_optimizer.eval_speed_up(
      plan_config=self.cur_plan_config,
      new_resources=new_resources,
    )
    
  def eval_best_effective_speed_up(self, 
                              new_resources: AcceleratorMesh2D, 
                              cur_plan_config: Optional[PlanConfig] = None,
                              **kwargs
  )-> Tuple[PlanConfig, float]:
    if cur_plan_config is None:
      cur_plan_config = self.cur_plan_config

    return self.job_optimizer.eval_best_effective_speed_up(
      cur_plan_config=cur_plan_config,
      new_resources=new_resources,
      **kwargs
    )
  
  """
  cancel
  """
  def cancel(self):
    self._handle_cancelled()
    
  """
  ff
  """
  def ready_for_ff(self) -> bool:
    # refuse when PENDING
    if self.status not in [JobStatus.WAITING, JobStatus.RUNNING]:
      return False
    # WAITING job always ok for ff
    if self.status == JobStatus.WAITING:
      return True
    
    with self.guard_lock:
      if not self.job_optimizer.est.initialized():
        return False
      
      # RUNNING job ok for ff if.f. iter_time has stabled
      now = time.time() # use wallclock time
      from Suika.cluster.scheduler import default_ff_time_margin
      window_start = now - default_ff_time_margin
      self.ff_infos = [
        ff_info for ff_info in self.ff_infos if window_start <= ff_info[0][0]
      ]
      
      if not len(self.ff_infos) or (self.ff_infos[0][0][0] - window_start) > default_ff_time_margin / 2: # ensure the job has been reporting statistics for a while
        return False
      
      min_iter_time = min([ff_info[2] for ff_info in self.ff_infos])
      max_iter_time = max([ff_info[2] for ff_info in self.ff_infos])
      if max_iter_time - min_iter_time < 0.05 * min_iter_time: # 5% variation
        return True
      else:
        return False
  
  def get_ff_completion_time(self) -> float:
    """
    Get the expected completion time given ff_infos
    This function is expected to be called when ready_for_ff() is True.
    """
    with self.guard_lock:
      
      n_iter = self.other_context.get("n_iter", -1)
      if n_iter <= 0:
        assert False, f"{self} n_iter not set when ff is enabled" 
        return -1
      
      (_, source_time), cur_iter, _ = self.ff_infos[-1]
      iter_time = sum(
        ff_info[2] for ff_info in self.ff_infos
      ) / len(self.ff_infos)
      
      completion_time = source_time + (n_iter - cur_iter) * iter_time
      # self.log(f"FF: n_iter {n_iter}, cur_iter {cur_iter},  iter_time: {iter_time:.3f}, cur_time {source_time:.2f}, completion {completion_time:.1f}")
      
      return completion_time    
  
  def fast_forward(self, target_time: float):
    """
    Issue the fast forward towards target_time (virtual)
    This function is expected to be called when ready_for_ff() is True, and all other checks done at scheduler side
    """
    with self.guard_lock:
      if self.status != JobStatus.RUNNING or not self.ff_infos:
        return False
      
      (_, source_time), cur_iter, _ = self.ff_infos[-1]
      iter_time = sum(
        ff_info[2] for ff_info in self.ff_infos
      ) / len(self.ff_infos)
      
      n_iter = self.other_context.get("n_iter", -1)
      if n_iter <= 0:
        assert False, f"{self} n_iter not set when ff is enabled" 
        return False
      
      if target_time + 1e-2 <= source_time:
        assert False, f"{self} ff failed: target_time {target_time} should be greater than source_time {source_time}"
        return False
      
      target_cur_iter = int((target_time - source_time) / iter_time) + cur_iter
      
      if target_cur_iter > n_iter:
        assert False, f"{self} ff failed: target_cur_iter {target_cur_iter} exceeds n_iter {n_iter}"
        return False
      
      self.job_server.broadcast((Job2WorkerMsg.FF, target_cur_iter))
      wallclock_now = time.time()
      virtual_now = target_time
      self.ff_infos.append(
        ((wallclock_now, virtual_now), target_cur_iter, iter_time) # ensure cur_iter is correct during continuous fast forward
      )
      
      return True
    
  """
  log
  """
  def log(self, msg: str, level: int = logging.DEBUG):
    if self.logger is not None:
      pattern_msg = f"[{self.job_name}] {msg}"
      self.logger.log(level, pattern_msg)
  
  
  """
  Guard to handle worker communication and status update
  """
  def guard(self):

    last_msg_time = {}
    timeout = 600 # in sec
    job_sleep_interval = 0.1
    now = time.time()

    wallclock_mismatch_warned = False

    self.log("guard process started")
    if not self.job_server.running:
      self.job_server.start()

    while True:
      
      # acquire control
      with self.guard_lock:
        if self.reconfig_thread is not None:
          self.reconfig_thread.join()
          self.reconfig_thread = None

        # recv from clients with polling
        ranks = list(self.job_server.connections.keys()) # save ranks to avoid dict change during iteration
        for i in ranks:
          msg_cnt = 10
          while msg_cnt > 0:
            msg_cnt -= 1
            
            obj = self.job_server.recv(i, wait = False)
            if obj is None: # no more message
              break

            msg_type, value = obj
            
            assert msg_type in Worker2JobMsg, f"Invalid message type {msg_type}"
            
            now = time.time()

            if msg_type == Worker2JobMsg.COMPLETED:
              self.log(f"worker {i} completed")
              self._handle_completed(i)
              # wait for all to complete, don't return immediately
            
            elif msg_type == Worker2JobMsg.FAILED:
              self.log(f"worker {i} failed", level=logging.ERROR)
              self._handle_failed(i)
              
            elif msg_type == Worker2JobMsg.SHRANK:
              self.log(f"worker {i} shrank")
              self._handle_shrank(i)
            
            elif msg_type == Worker2JobMsg.HEARTBEAT:
              # self.log(f"worker {i} heartbeat")
              pass
            
            elif msg_type == Worker2JobMsg.PROFILE:
              assert isinstance(value, Dict), f"Invalid profile data of type {type(value)}"
              # self.log(f"worker {i} profile updated")
              if i == 0:
                self.job_optimizer.est.update_constants_from_profile(
                  C_raw = value, 
                  plan_config=self.cur_plan_config
                )
                if not self.job_optimizer.est.block_update:
                  self.record_iter_time(value.get("iter_time", -1.0))
              
            elif msg_type == Worker2JobMsg.RECONFIG_START:
              self.log(f"worker {i} reconfig start")
              if i == 0 and len(self.reconfig_start_timers) > 0:
                timer = self.reconfig_start_timers.pop(0)
                timer()
            
            elif msg_type == Worker2JobMsg.RECONFIG_COMPLETED:
              self.log(f"worker {i} reconfig complete")
              self._handle_reconfig_complete(i)          
              
              if i == 0 and len(self.reconfig_stop_timers) > 0:
                timer = self.reconfig_stop_timers.pop(0)
                timer()
                
            elif msg_type == Worker2JobMsg.FF:
              
              # self.log(f"worker {i} fast forwarding info ({obj}) received")
              if i == 0 and self.status == JobStatus.RUNNING:
                ff_wallclock_time, ff_cur_iter, ff_iter_time = value
                if abs(ff_wallclock_time - now) > 5:
                  if not wallclock_mismatch_warned:
                    self.log(f"ff_wallclock_time differs from scheduler wallclock time at receiving too much ({ff_wallclock_time - now:.1f} sec); are your machines' clocks mismatching? fall back to scheduler wallclock time.", level=logging.WARNING)
                    wallclock_mismatch_warned = True
                  ff_wallclock_time = now
                
                ff_virtual_time = self.scheduler_timer()
                latest_cur_iter = self.ff_infos[-1][1] if len(self.ff_infos) else -1 # ensure monotonicity
                if ff_cur_iter >= latest_cur_iter:
                  self.ff_infos.append(((ff_wallclock_time, ff_virtual_time), ff_cur_iter, ff_iter_time))
              
            else:
              assert False, f"Invalid message type {msg_type}"
              
          if i in last_msg_time and now - last_msg_time[i] > timeout:
            self.log(f"worker {i} timeout")
            self._handle_failed(i)
          last_msg_time[i] = now

        try:
          # update worker status, and update job status if needed
          
          # remove shrank workers from worker status 
          shrank_ranks = [
            k for k, v in self.worker_status.items() if v == WorkerStatus.SHRANK
          ]
          for rank in shrank_ranks:
            self.job_server.close_connection(rank)
            if rank in last_msg_time:
              del last_msg_time[rank]
            
          self.worker_status = {
            k: v for k, v in self.worker_status.items() if v != WorkerStatus.SHRANK
          }
          
          # complete
          # keep broadcasting completion information to avoid hang
          if any([v == WorkerStatus.COMPLETED for v in self.worker_status.values()]):
              self.job_server.broadcast(
                (Job2WorkerMsg.COMPLETED, None)
              )
          
          # check if all workers are completed
          # to COMPLETED
          not_pending_worker_status = [
            v for v in self.worker_status.values() if v != WorkerStatus.PENDING
          ]
          if len(not_pending_worker_status) > 0 and \
            all([v == WorkerStatus.COMPLETED for v in not_pending_worker_status]) and self.status != JobStatus.COMPLETED:
            # remark: len > 0 is needed, or at the moment all workers are shrank due to checkpoint-resume, or preemption, it should not be considered completed
            self.status = JobStatus.COMPLETED
            self.job_server.broadcast(
              (Job2WorkerMsg.ACK, None)
            )
            self.log(f"guard: job completed")
            break
          
          # reconfig_complete
          # to RUNNING
          if len(self.worker_status) > 0 and \
            all([v == WorkerStatus.RUNNING for v in self.worker_status.values()]) and self.status != JobStatus.RUNNING:
            self.status = JobStatus.RUNNING
            self.job_optimizer.est.set_block_update_constants(False)
            self.log(f"guard: job running")

          # to WAITING (all workers shrank, preempted)
          if not len(self.worker_status):
            self.status = JobStatus.WAITING

          # to FAILED
          if any([v == WorkerStatus.FAILED for v in self.worker_status.values()]) and self.status != JobStatus.FAILED:
            self.status = JobStatus.FAILED
            self.log(f"guard: job failed")
            

        except RuntimeError as e:
          self.log(f"{e}")
          self.log(f"error during iterating worker status. likely dict size changed due to reconfig. ignored")
          # after adding guard lock, this should not happen, raise e if it does   
          raise e
        
        # guard exits if job ends (or preempted)
        if self.status not in [JobStatus.PENDING, JobStatus.RUNNING]:
          self.job_server.stop()
          return

        # yield control to reconfig if needed

      # sleep outside the Lock context
      time.sleep(job_sleep_interval)
      

      
  def _handle_failed(self, rank: Optional[int] = None):
    if rank is not None:
      self.worker_status[rank] = WorkerStatus.FAILED
    else:
      self.status = JobStatus.FAILED
    # note that FAILED status is not fully tracked, and the EXIT is sent without waiting for response.
    # we expect no FAILURE during entire experiment.
    self.job_server.broadcast(
      (Job2WorkerMsg.EXIT, None)
    )
    return
  
  def _handle_completed(self, rank: int):
    # if self.worker_status[rank] not in [WorkerStatus.RUNNING, WorkerStatus.COMPLETED]:
    #   self.log(f"worker {rank} completed but is not in RUNNING or COMPLETED, got ({self.worker_status[rank]}), ignored; however such workers are not expected to send complete msg, check your code", level=logging.WARNING)
    #   return
    
    self.worker_status[rank] = WorkerStatus.COMPLETED

    # following JobStatus change is handled by guard at the end of each round of polling
    return
  
  def _handle_cancelled(self):
    self.status = JobStatus.CANCELLED
    
    self.job_server.broadcast(
      (Job2WorkerMsg.EXIT, None)
    )
    return
  
  def _handle_reconfig_complete(self, rank):
    if self.worker_status[rank] != WorkerStatus.PENDING:
      self.log(f"worker {rank} reconfig complete but is not in PENDING, got ({self.worker_status[rank]}), ignored; however such workers are not expected to send reconfig complete msg, check your code", level=logging.WARNING)
      return
    
    self.worker_status[rank] = WorkerStatus.RUNNING
    
    # JobStatus change is handled by guard at the end of each round of polling
    return
  
  def _handle_shrank(self, rank):
    if self.worker_status[rank] not in [WorkerStatus.RUNNING, WorkerStatus.PENDING]:
      self.log(f"worker {rank} shrank but is not in RUNNING or PENDING, got ({self.worker_status[rank]}), ignored; however such workers are not expected to send shrank msg, check your code", level=logging.WARNING)
      return
    
    self.worker_status[rank] = WorkerStatus.SHRANK
    self.job_server.send(
      dst=rank,
      obj=(Job2WorkerMsg.ACK, None)
    )
    
    # harvest of worker_stats[rank] is handled by guard at the end of each round of polling
    
    
  @staticmethod
  def _recursive_launch(launch_func, rank_mesh, acc_mesh, hostname_mesh, *args, **kwargs):
  
    if not isinstance(acc_mesh, list):
      launch_func(acc_mesh, rank_mesh, hostname_mesh, *args, **kwargs)
      return
    
    for sub_acc, sub_rank, sub_host in zip(acc_mesh, rank_mesh, hostname_mesh):
      Job._recursive_launch(launch_func, sub_rank, sub_acc, sub_host)

  # printing
  def __str__(self):
    return f"Job({self.job_name}, {self.job_id}, {self.status.name})"
  def __repr__(self):
    return self.__str__()
  
  # hashing (job_name, job_id)
  def __hash__(self):
    return hash((self.job_name, self.job_id))
  def __eq__(self, other):
    if not isinstance(other, Job):
      return False
    return self.job_name == other.job_name and self.job_id == other.job_id
  
"""
These methods should be used by worker process
"""

"""
@EXPORT
connect to job manager of scheduler (default: read environ JOB_ENDPOINT)

if strict is set, assert connection success

"""
def connect_to_scheduler(
  rank: Optional[str] = None, 
  endpoint: Optional[str] = None, 
  logger: Optional[Logger] = None, 
  strict: bool = False,
  log_func: Optional[Callable] = None,
  ) -> SuikaSocketClient:
  """
  Connect to the cluster scheduler.
  """
  if log_func is None:
    log_func = lambda x: None
      
  rank = os.environ.get("RANK", None) if rank is None else int(rank) # note that Client requires id to be int
  job_endpoint = os.environ.get("JOB_ENDPOINT", None) if endpoint is None else endpoint
  if rank == None:
    raise ValueError("RANK not set")  
  if job_endpoint == None:
    raise ValueError("JOB_ENDPOINT not set")
  
  rank = int(rank)
  log_func(f"({rank}) connect to scheduler at {job_endpoint}")
  job_client = SuikaSocketClient(rank, job_endpoint, 
                                 # logger=logger
                                )
  
  success = job_client.start()
  if strict:
    assert success, f"Failed to connect to scheduler at {job_endpoint}"
  log_func(f"connect to scheduler {'succeeded' if success else 'failed'}")
  
  return job_client

# utility to wait for ack message
def wait_ack(job_client, wait_ack_timeout = 60):
  wait_ack_now = time.time()
  wait_ack_ddl = wait_ack_now + wait_ack_timeout
  while (wait_ack_now < wait_ack_ddl and job_client.running):
    wait_ack_now = time.time()
    obj = job_client.recv(wait=False)
    if obj is None:
      time.sleep(wait_ack_timeout / 100)
      continue
    elif obj[0] == Job2WorkerMsg.ACK:
      break

"""
@EXPORT

Report profile to scheduler and check if reconfiguration is needed.
Also handle cancel logic and completion logic.

The background deamon function for worker's suika_train

Reason to set it async:
  avoid hanging if reconfig and completion overlap.
  (where part of works enter train_save, and the others end up in completed)
  
  though we added dist_agree in suika_train, which theoretically should be enough; we add a double protection here. 
"""
def communicate_with_scheduler_agent(
  job_client: SuikaSocketClient, 
  shared_dict: Dict[str, Any],
  ready: InterProcessFlag,
  signal_queue: queue.Queue,
  lock: threading.Lock,
  log_func: Optional[Callable] = None,
  ) -> bool:
  if log_func is None:
    log_func = lambda x: None  
  
  agent_sleep_interval = 0.2

  agent_send_interval = 1
  agent_send_last = 0

  agent_exit_delay = 1
  agent_exit_ddl = None
  while job_client.running and \
    (agent_exit_ddl is None or now < agent_exit_ddl):
    now = time.time()

    if ready.check("agent_exit") and agent_exit_ddl is None:
      agent_exit_ddl = now + agent_exit_delay # add a marginal delay for COMPLETE message to arrive

    # send profile to scheduler      
    if now - agent_send_last >= agent_send_interval:
      profile = shared_dict.get("profile", None)
      if profile is not None: 
        assert isinstance(profile, Dict), f"perfile must be dict, got {type(profile)}"
        log_func(f"send profile to scheduler: {profile}")
        msg = (Worker2JobMsg.PROFILE, profile)
        
        cur_iter = shared_dict.get("cur_iter", -1)
        iter_time = shared_dict.get("iter_time", -1)
        log_func(f"ff_info: cur_iter {cur_iter} / iter_time {iter_time:.2f}")
        ff_info = (Worker2JobMsg.FF, (now, cur_iter, iter_time)) if cur_iter >= 0 and iter_time >= 0 else None
        
        shared_dict["iter_time"] = -1
        shared_dict["profile"] = None
      else:
        msg = (Worker2JobMsg.HEARTBEAT, None)
        ff_info = None
        agent_send_last = now
      
      job_client.send(msg)
      
      if ff_info is not None:
        job_client.send(ff_info)

    # receive message
    obj = job_client.recv(wait = False)
    if obj is None:
      time.sleep(agent_sleep_interval)
    else:
      with lock:
        msg_type, value = obj
        if msg_type == Job2WorkerMsg.EXIT:
          log_func(f"recv EXIT from scheduler")
          
          # job_client.send((Worker2JobMsg.EXITED, None))
          
          kill_all_child_processes()
          exit(1) # TODO: Allow graceful quit... leave for later
          
        elif msg_type == Job2WorkerMsg.COMPLETED:
          log_func(f"recv COMPLETED from scheduler")

          log_func(f"agent: set and send COMPLETED")
          job_client.send((Worker2JobMsg.COMPLETED, None))
          ready.set("recv_complete") # set recv_complete flag before wait for ack.

          log_func(f"agent: before wait ack")
          wait_ack(job_client)
          log_func(f"agent: after wait ack")

          s, f = kill_processes(shared_dict["worker_procs"], recursive=True)
          log_func(f"agent: procs to kill {shared_dict['worker_procs']}, success {s}, failure {f}")

          exit(0)
        elif msg_type == Job2WorkerMsg.RECONFIG:
          
          new_plan_config, env = value
          assert isinstance(new_plan_config, PlanConfig) or new_plan_config is None, f"new_plan_config must be PlanConfig or None for preemption, got {type(new_plan_config)}"
          cur_plan_config = shared_dict.get("plan_config", None)
          if cur_plan_config == new_plan_config:
            continue
          log_func(f"recv RECONFIG from scheduler")
          shared_dict["plan_config"] = new_plan_config
          shared_dict["env"] = env
          log_func(f"new_plan_config:\n{new_plan_config}")
          log_func(f"env at recv:\n{env}")
          
          signal_queue.put(True)

        elif msg_type == Job2WorkerMsg.CHECKPOINT_RESUME:
          # job marked as checkpoint_resume
          if shared_dict.get("checkpoint_resume", -1) != -1:
            continue
          log_func(f"recv CHECKPOINT_RESUME from scheduler")

          checkpoint_path = value
          shared_dict["checkpoint_resume"] = checkpoint_path
          
          log_func(f"checkpoint path: {checkpoint_path}")
        elif msg_type == Job2WorkerMsg.ACK:
          log_func(f"recv ACK from scheduler")
          # ack for completed / failed / shrank
          ready.set("scheduler_ack")

        elif msg_type == Job2WorkerMsg.FF:
          # fast forward
          log_func(f"recv FF from scheduler")
          target_cur_iter = value
          shared_dict["ff_iter"] = target_cur_iter
          log_func(f"set ff_iter to {target_cur_iter}")
        else:
          raise ValueError(f"Invalid Job2WorkerMsg type {msg_type}")
      time.sleep(agent_sleep_interval)

  if not job_client.running:
      log_func(f"agent exit on job_client error")
    
"""
@EXPORT

Check if reconfig is received from scheduler

Combined with communicate_with_scheduler_agent
"""
def reconfig_received_from_scheduler(signal_queue) -> Union[str, None]:
  """
  Check if reconfig is received from scheduler
  """
  try:
    msg = signal_queue.get_nowait()
    assert msg is True, f"Invalid signal {msg}"
    return msg
  except queue.Empty:
    return False
  
  
  
"""
Detailed Flow Chart of Worker Status Change During Reconfiguration

(I) if migration:

Job.reconfig():
  JobStatus -> PENDING
  
  launch_new_workers: WorkerStatus -> PENDING -> RUNNING
  
  send_reconfig_func: WorkerStatus -> PENDING
  
  return

[async] Job.guard(): 
  recv Worker2JobMsg.RECONFIG_COMPLETE -> _handle_reconfig_complete:
      WorkerStatus -> RUNNING
      
  recv Worker2JobMsg.SHRANK -> _handle_reconfig_complete:
      WorkerStatus -> SHRANK

  del_shrank_workers
      if all workers are completed:
        JobStatus -> COMPLETED
      if all workers are running:
        JobStatus -> RUNNING

(II) if checkpoint resume:

Job.reconfig():
  JobStatus -> PENDING
  
  send_reconfig_func (to inform previous workers to exit):
    WorkerStatus -> PENDING
    
  wait JobStatus till RUNNING
  [async] Job.guard() recv Worker2JobMsg.SHRANK -> _handle_reconfig_complete -> RUNNING:
  
  if complete preemption:
    JobStatus -> WAITING
    return
  else:
    JobStatus -> PENDING
  
  launch new worker: WorkerStatus -> PENDING -> RUNNING
  send_reconfig_func: WorkerStatus -> RUNNING

  return
  
  [async] Job.guard() recv Worker2JobMsg.RECONFIG_COMPLETE -> _handle_reconfig_complete -> RUNNING  
  
"""