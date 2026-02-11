from logging import Logger
from typing import Callable, Optional, Dict, Tuple, Any

import time
import os

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import threading
import queue

from Suika.cluster.job import Worker2JobMsg, connect_to_scheduler, communicate_with_scheduler_agent, reconfig_received_from_scheduler, wait_ack
from Suika.profiler import Profiler, DummyProfiler
from Suika.cluster.commu import InterProcessFlag
from Suika.utils import dist_barrier, dist_agree, kill_processes
from Suika.migration import dummy_migration_IPC_state_dict

# sleep hyper parameters
reconfig_sleep_interval = 0.5


KwArgs = Dict[str, Any]  

class KwArgPlaceholder:
  name: Optional[str] # None means 'infer'
  def __init__(self, name: Optional[str] = None):
    self.name = name
  
def create_kwargs_template(func: Callable) -> KwArgs:
  """
  Get a dict of KwArgPlaceholder for the function's kwargs.
  The keys are the argument names, and the values are KwArgPlaceholder.
  """
  import inspect
  sig = inspect.signature(func)
  return {k: KwArgPlaceholder(name=k) for k in sig.parameters.keys() if k != "self"} # exclude self if exists
  
"""
Auxiliary functions for suika_train
Defined in global scope as pickle cannot handle local method
"""  
def materialize_kwargs(virtual_kwargs, shared_kwargs: KwArgs, local_kwargs: KwArgs) -> KwArgs:
    kwargs = {}
    for k, v in virtual_kwargs.items():
        if isinstance(v, KwArgPlaceholder):
            karg_name = v.name if v.name is not None else k
            # assert karg_name in shared_kwargs or karg_name in local_kwargs, f"Missing key {karg_name} in shared_kwargs / local_kwargs"
            if karg_name in local_kwargs:
              kwargs[k] = local_kwargs.get(karg_name, None)
            else:
              kwargs[k] = shared_kwargs.get(karg_name, None)
            
        else:
            kwargs[k] = v
    return kwargs

def exec_task(task: Tuple[Callable, KwArgs], shared_kwargs: KwArgs, local_kwargs: KwArgs):
    func, virtual_kwargs = task
    kwargs = materialize_kwargs(virtual_kwargs, shared_kwargs, local_kwargs)
    
    result = func(**kwargs)
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    
    if result is not None:
      assert isinstance(result, tuple) and len(result) == 2, \
        f"Function {func.__name__} should return Tuple[Dict, Dict], but got {type(result)}"
      # assume to be Tuple[Dict, Dict]
      if result[0]:
        shared_kwargs.update(result[0])
      if result[1]:
        local_kwargs.update(result[1])

# only difference to exec_task is that it detects StopIteration
def exec_train_step_task(train_step_task: Tuple[Callable, KwArgs], shared_kwargs: KwArgs, local_kwargs: KwArgs) -> bool:
    func, virtual_kwargs = train_step_task
    kwargs = materialize_kwargs(virtual_kwargs, shared_kwargs, local_kwargs)
    
    try:
        result = func(**kwargs)
    except StopIteration:
        return False
    if torch.cuda.is_available():
      torch.cuda.synchronize()

    if result is not None:
      assert isinstance(result, tuple) and len(result) == 2, \
        f"Function {func.__name__} should return Tuple[Dict, Dict], but got {type(result)}"
      if result[0]:
        shared_kwargs.update(result[0])
      if result[1]:
        local_kwargs.update(result[1])
      
    return True

  

def train_proc_target(
  prepare_context_task: Tuple[Callable, KwArgs],
  train_load_task: Tuple[Callable, KwArgs],
  train_step_task: Tuple[Callable, KwArgs],
  train_save_task: Tuple[Callable, KwArgs],
  shared_kwargs: "mp.managers.DictProxy",
  IPC_queues: mp.Queue,
  ready: InterProcessFlag,
  log_func: Callable[[str], None]
  ):
    # put migration queue into local_kwargs, as it cannot be transferred via mp.Manager
    local_kwargs: KwArgs = {"IPC_queues": IPC_queues}
    
    # Prepare context
    if ready.check("exit") or ready.check("recv_complete"): 
      # unfortunately, we cannot use dist_agree before entering prepare context. 
      # however, unlike following case, some client fails to join init_dist won't trigger immediate abort
      # this leaves time for COMPLETE message from scheduler to arrive
      return 0
    log_func(">before prepare_context")
    exec_task(prepare_context_task, shared_kwargs, local_kwargs)
    log_func("<after prepare_context")
    
    log_func("set reconfig_prepare")
    ready.set("reconfig_prepare")

    # Wait for the previous process to enter train_save
    log_func(">before wait guard_reconfig_ack")
    if not dist_agree(ready.wait("guard_reconfig_ack", exit_flag = ["exit", "recv_complete"])):
       return 0
    log_func("<after wait guard_reconfig_ack")

    log_func("reset all flags")
    ready.reset_all(exclude = ["guard_reconfig_save", "exit", "recv_complete"])

    # Train 
    log_func(">before train_load")
    exec_task(train_load_task, shared_kwargs, local_kwargs)
    log_func("<after train_load")

    ready.set("reconfig_load")

    # Wait for the previous process to exit
    log_func(">before wait guard_reconfig_save")
    if not dist_agree(ready.wait("guard_reconfig_save", exit_flag = ["exit", "recv_complete"])):
      return 0    
    ready.reset("guard_reconfig_save")
    log_func("<after wait guard_reconfig_save")


    # Enter normal training phase (where next reconfig interrupt is acceptable)
    dist_barrier()

    # Start training
    # train_step
    is_first_step = True
    profiler = Profiler()
    profiler.start("iter_time")

    log_func("train_step")
    while exec_train_step_task(train_step_task, shared_kwargs, local_kwargs):
        # Check if need to reconfigure
        reconfig_prepare = ready.check("reconfig_prepare")
        if dist_agree(reconfig_prepare):
            log_func("reconfig_prepare / set reconfig_ack")
            ready.set("reconfig_ack")
            # train_save
            log_func(">before train_save")
            exec_task(train_save_task, shared_kwargs, local_kwargs)
            log_func("<after train_save")
            return 0
        elif is_first_step:
            # ensure that optimizer is instantiated before reconfig, otherwise migration may fail
            log_func("set interruptable")
            ready.set("interruptable")
            is_first_step = False
        
        # Record statistic for fast forward
        profiler.stop("iter_time")
        iter_time = profiler.report("iter_time")
        if iter_time is not None:
          shared_kwargs["iter_time"] = iter_time  
        profiler.start("iter_time")

    if dist.is_initialized():
       dist.barrier()
       dist.destroy_process_group()
       
    # if reconfig happens at when training end, ensure they exit
    log_func("target exit")
    ready.set("exit")
    return 0

"""
@EXPORT

  @name: suika_train
  @brief:
    Integrated adaptive training for Suika.

    Procedure:
      - Call prepare_context.
      - Call train_load.
      - While train_step does not raise StopIteration:
          - If reconfig_event:
              - Call train_save.
              - Exit (switch to next process).
      - Call train_save
      - Exit (train complete).

    The actual procedure is managed across multiple processes.

  @args:
    1. prepare_context_task: Tuple[Callable, KwArgs]
       (prepare_context_func, prepare_context_kwargs)
       Function to prepare the training context.

    2. train_step_task: Tuple[Callable, KwArgs]
       (train_step_func, train_step_kwargs)
       Atomic train step function that cannot be interrupted.
       Raise StopIteration exception to stop training.

    3. train_load_task: Tuple[Callable, KwArgs]
       (train_load_func, train_load_kwargs)
       Load function to initialize model, dataloader, etc.

    4. train_save_task: Tuple[Callable, KwArgs]
       (train_save_func, train_save_kwargs)
       Save function to save model, dataloader, etc.
       Coupled with train_load_task.

    5. shared_kwargs: Optional[KwArgs]
       Initial values of shared kwargs storage for KwArgPlaceholder.
    
    6. log_func: Optional[Callable]
       Logging function to log messages.

    7. profiler: Optional[Profiler]
       Profiler to profile the training process.
       
  @return:
    shared_kwargs: KwArgs

  @remark:
    1. Functions should refer to karg "plan_config / migration_kwargs" to support adaptive training.
       - plan_config is initialized/updated from the cluster scheduler.
      - IPC_queues: Tuple[mp.Queue, mp.Queue] are set by guard process for migration_save/load
      - checkpoint_resume: Optional[str], if not None, use traditional checkpoint resume instead of migration; the argument is path to checkpoint dir.
      
               
    2. All functions must return Tuple[dict, dict], which will be merged into shared_kwargs and local kwargs.
      - The first dict is merged into shared_kwargs, and the second dict is merged into local_kwargs.
        - shared_kwargs persists reconfiguration, but not across ranks.
        - local_kwargs do not persiss reconfiguration. (typically used to pass model and optimizer across load/train_step/save)
      - "profile" field in shared_dict should be set by train_step_func, which will be sent to scheduler.
       
    3. Use KwArgPlaceholder to define a placeholder in func's kwargs,
       which will be replaced by value in shared_kwargs before call.
       
    4. shared_kwargs is also returned from this function, so statistical data can be saved to shared_kwargs for further use.
    
    5. shared_kwargs is per rank process shared, but not shared across ranks!
    
  @flags:
    1. reconfig_prepare: new process has finished context preparation
    2. reconfig_ack: previous process has acknowledged reconfig and enters train_save (so next process can proceed to train_load)
    3. reconfig_save: previous process has finished saving (deprecated)
    4. training: job is in training phase (where next reconfig interrupt is acceptable)
    5. exit: job is exiting (all process should exit)
    6. guard_reconfig_ack: guard process have received reconfig_ack (avoided deadlock in main .wait(reconfig_ack))
    6. guard_reconfig_save: guard process have confirmed previous process has joined (avoid two processes competing for GPU memory)
  
  flow of flags:

  next  | reconfig_prepare |                                   | (train_load) |                              | continue to (train_step)
  prev  |                  | reconfig_ack |                      (train_save) | (exit) |                    
  guard |                                 | guard_reconfig_ack |                       | guard_reconfig_save |

  state transition in "next/prev" uses dist_agree whenever possible.
  a redundant 'wait' is also added to accept completion message from scheduler, further reducing the possiblibity of killing NCCL resulting in task FAILED

  @note:
    remember to distroy process group in train_save_func


"""

def suika_train(
    prepare_context_task: Tuple[Callable, KwArgs],
    train_load_task: Tuple[Callable, KwArgs],
    train_step_task: Tuple[Callable, KwArgs],
    train_save_task: Tuple[Callable, KwArgs],
    shared_kwargs: KwArgs = {},
    logger: Optional[Logger] = None,
    profiler: Optional[Profiler] = None,
  ) -> KwArgs:
  # preprocess arguments
  if logger is None:
    log_func = lambda msg: None
  else:
    log_func = logger.info
  if profiler is None:
    profiler = DummyProfiler()
  
  # create connection to cluster
  job_client = connect_to_scheduler(
    rank=None, 
    endpoint=None,
    logger=logger,
    log_func=log_func,
  )
  
  # multiprocess context
  # if context not set, set start_method to spawn
  if not mp.get_start_method(allow_none=True):
    mp.set_start_method("spawn")
  
  with mp.Manager() as mp_manager:
    
    # prepare migration_queue for migration
    IPC_queues = (mp.Queue(), mp.Queue())
    rank = int(os.environ["RANK"])
    if rank == 0:
      IPC_queues[0].put(0) # dummy world_size
    IPC_queues[0].put(dummy_migration_IPC_state_dict)
    
    # prepare sharded kwargs
    shared_kwargs = mp_manager.dict(shared_kwargs)
    shared_kwargs["checkpoint_resume"] = -1 # cluster will send None
    shared_kwargs["worker_procs"] = []
    
    # set initial state to ready, as no previous process exists
    ready = InterProcessFlag(mp_manager)
    ready.set("interruptable")
    
    # start agent thread
    signal_queue = queue.Queue()
    agent_lock = threading.Lock()
    agent_thread = threading.Thread(target=communicate_with_scheduler_agent,
                                    kwargs = {
                                      "job_client": job_client,
                                      "shared_dict": shared_kwargs,
                                      "ready": ready,
                                      "signal_queue": signal_queue,
                                      "lock": agent_lock,
                                      "log_func": log_func,
                                    })
    agent_thread.daemon = True
    agent_thread.start()
    
    """
    Actual Process Management
    """
    args = [
      prepare_context_task,
      train_load_task,
      train_step_task,
      train_save_task,
      shared_kwargs, 
      IPC_queues,
      ready,
      log_func
    ]
    
    def start_proc():
      # Setup new process
      env = shared_kwargs.get("env", {})
      # assert env, "Environment variables not set in shared_kwargs"
      log_func(f"env at start_proc: \n{env}")
      os.environ.update(env)
      proc = mp.Process(target=train_proc_target, args = args)
      proc.start()
      shared_kwargs["worker_procs"] += [proc.pid]
      return proc
    
    proc = None
    
    # Main monitoring loop
    while proc is None or proc.is_alive():
      
      if reconfig_received_from_scheduler(signal_queue=signal_queue):
        log_func("suika_train: reconfig received from scheduler")
        # Reconfiguration received

        # Ensure last reconfig is complete
        log_func(">before wait interruptable")
        if not ready.wait("interruptable", exit_flag = ["exit", "recv_complete"]):
          log_func(">break wait interruptable")
          break
        # Reset intterruptable flag, reject any more reconfig
        log_func(">after wait interruptable")
        ready.reset("interruptable")
        
        # Determine if should_shrink
        new_world_size = int(shared_kwargs["env"]["WORLD_SIZE"])
        should_shrink = rank >= new_world_size

        # Determine if checkpoint resume is used
        is_checkpoint_resume = shared_kwargs["checkpoint_resume"] is not None

        
        # Case1: 
        #    suika migration is used, and the process is present in the next plan
        # Case2: 
        #    traditional checkpoint resume is used, and the proc is None

        should_launch_new_proc = (not is_checkpoint_resume and not should_shrink) or (is_checkpoint_resume and proc is None)
        log_func(f"suika_train: should_shrink {should_shrink}, is_checkpoint_resume {is_checkpoint_resume}, should_launch_new_proc {should_launch_new_proc}")
        if should_launch_new_proc:
          # Start new training process
          with agent_lock: # ensure new proc is seen by agent, at recv COMPLETE and killed.
            if ready.check("recv_complete") or ready.check("exit"):
              break
            prev_proc = proc
            proc = start_proc()
          
          if prev_proc is not None:
            log_func("new_proc launched; reconfig initiated")
          
            # Wait for prev process to join (complete pending train_step and train_save)
            profiler.start("reconfig-no-overlap")
            log_func(">before wait reconfig_ack")
            if not ready.wait("reconfig_ack", exit_flag = ["exit", "recv_complete"]):
              # previous process failed or completed
              # ! NOTE: must set proc = prev_proc, or the tail process to send COMPLETED or SHRANK would be wrong
              log_func("<break reconfig_ack")
              kill_processes([proc.pid], True)
              proc = prev_proc
              break
            log_func("<after wait reconfig_ack")

            ready.set("guard_reconfig_ack")
            profiler.start("reconfig")
            
            # Notify scheduler of reconfig start
            job_client.send((Worker2JobMsg.RECONFIG_START, None))

            # Wait for previous process to join
            log_func(">before prev_proc.join")
            prev_proc.join()
            with agent_lock:
              shared_kwargs["worker_procs"] = shared_kwargs["worker_procs"][1:] # hack: .pop(0) does not work for IPC dict
            log_func("<after prev_proc.join")
               
            if prev_proc.exitcode != 0:
              # previous process failed
              log_func("<break prev_proc.join")
              kill_processes([proc.pid], True)
              proc = prev_proc
              break
            
            profiler.stop("reconfig")
            profiler.stop("reconfig-no-overlap")
            log_func("prev_proc joined; reconfig complete")
          else:
            
            log_func(">before wait reconfig_prepare")
            if not ready.wait("reconfig_prepare", exit_flag=["exit", "recv_complete"]):
              log_func("<break wait reconfig_prepare")
              break
            log_func("<after wait reconfig_prepare")
            ready.set("guard_reconfig_ack")

            job_client.send((Worker2JobMsg.RECONFIG_START, None))
            log_func("prev_proc is None; reconfig complete")
            
          # Reconfig complete, notify scheduler
          ready.wait("reconfig_load", exit_flag=["exit", "recv_complete"])

          job_client.send((Worker2JobMsg.RECONFIG_COMPLETED, None))
          ready.set("guard_reconfig_save")
        else:
          # This process should be shrank after attending last migration_save
          ready.set("reconfig_prepare")
          ready.set("guard_reconfig_ack")
          
          # # case 1: checkpoint resume, complete preempted to 0 gpu - send RECONFIG_COMPLETE, as no following proc will do that
          # # case 2: checkpoitn resume, reconfig but not to 0 gpu - following proc on rank 0 will do that
          # # case 3: suika migration, shrink - following proc on rank 0 will do that
          # if new_world_size == 0:
          #   job_client.send((Worker2JobMsg.RECONFIG_COMPLETED, None))
          # break
      
      time.sleep(reconfig_sleep_interval)
      
    proc.join()

    ready.set("agent_exit")
    agent_thread.join()

    # Process has completed
    if not ready.check("recv_complete"): # normal exit

      if proc.exitcode != 0:
        job_client.send((Worker2JobMsg.FAILED, None))
        log_func(f"Training process exited with error code {proc.exitcode}")
      else:
        # notify scheduler of completion
        if ready.check("exit"):
          job_client.send((Worker2JobMsg.COMPLETED, None))
          log_func("Training process completed successfully")
        else:
          job_client.send((Worker2JobMsg.SHRANK, None))
          log_func("Training process shrank successfully")
      
      # at this point agent has exited, so it's safe to recieve message from main guard thread
      if not ready.check("scheduler_ack"):
        wait_ack(job_client)
      
    else: # sheduler instructed completion
       log_func(f"Received complete from scheduler, ignoring training process exit code {proc.exitcode}")

    s, f = kill_processes(shared_kwargs["worker_procs"], True)
    log_func(f"suika_train: procs to kill {shared_kwargs['worker_procs']}, success {s}, failure {f}")

    log_func("Agent process joined")
    # log_func(f"{profiler.full_report_str(require_time=True, require_memory=False)}")
        
  return shared_kwargs
