import logging
import torch
import os
from typing import Any, List, Tuple, Union, Optional, Iterable

SUIKA_LOG_DIR = os.getenv("SUIKA_LOG_DIR", "./log")
def set_log_dir(log_dir: str):
  global SUIKA_LOG_DIR
  SUIKA_LOG_DIR = log_dir

def get_logger(name: str, level: int = logging.DEBUG, console_level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(formatter)
    import os
    file_path = f"{SUIKA_LOG_DIR}/{name}.log"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    fh = logging.FileHandler(file_path, mode='w')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

from torch.nn import Module


def print_state_dict(model: Module, logger: logging.Logger = None) -> None:
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        logger.info(f"{k}: {v.size()}")
        
def print_profile(model: Module, logger: logging.Logger = None) -> None:
  logger.info(f"{model.prof.full_report_str()}")
  
# flatten nested list
def flatten_mesh(mesh: List):
  if not isinstance(mesh, List):
    return [mesh]
  
  flattened = []
  for sub_mesh in mesh:
    flattened.extend(flatten_mesh(sub_mesh))
  
  return flattened

"""
Check is power of two
"""

def is_power_of_2(n: int) -> bool:
    """
    Check if a number is a power of 2.

    Args:
        n (int): The number to check.

    Returns:
        bool: True if n is a power of 2, False otherwise.
    """
    return (n & (n - 1)) == 0 and n != 0


"""
change device context manager
"""
from contextlib import contextmanager
@contextmanager
def set_default_device_context(device: Union[str, torch.device]):
  """
  Context manager to set the default device for PyTorch operations.

  Args:
      device (Union[str, torch.device]): The device to set as default.
  """
  try:
    if device is not None:
      original_device = torch.get_default_device()
      torch.set_default_device(device)
    yield
  finally:
    if device is not None:
      torch.set_default_device(original_device)

  
"""
distributed barrier
"""
import torch.distributed as dist
def dist_barrier():
  # if not initialize, do nothing
  if dist.is_initialized():
    dist.barrier()

def broadcast_objects(*args, group = None) -> Any:
  broadcast_list = list(args)
  torch.distributed.broadcast_object_list(broadcast_list, src=0, group=group)
  if len(broadcast_list) == 1:
    return broadcast_list[0]
  else:
    return iter(broadcast_list)
  
def dist_agree(flag: bool, group = None) -> bool:
  flags = [flag] * dist.get_world_size()
  dist.all_gather_object(flags, flag, group=group)
  return all(flags)


"""
kill and clean up
"""
import subprocess
import psutil
import os

def is_mps_daemon_running() -> bool:
  try:
    subprocess.check_output(['pgrep', '-f', 'nvidia-cuda-mps-control'])
    return True
  except subprocess.CalledProcessError:
    return False

def get_mps_server_id(pid) -> int:
  if not is_mps_daemon_running():
    return -1
  cmd = 'echo ps | nvidia-cuda-mps-control'
  result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
  clients = {}
  for line in result.stdout.strip().splitlines():
      if line.startswith('PID'):
          continue  # skip header
      parts = line.strip().split()
      if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
          client_pid = int(parts[0])
          server_pid = int(parts[2])
          clients[client_pid] = server_pid
  if pid in clients:
    return clients[pid]
  else:
    return -1
  
def mps_safe_kill(pid: int) -> bool:
  client_pid = pid
  server_pid = get_mps_server_id(pid)
  if server_pid >= 0:
    cmd = f'echo "terminate_client {server_pid} {client_pid}" | nvidia-cuda-mps-control'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
      return False
  
  try:
    proc = psutil.Process(pid)
    proc.kill()
    return True
  except psutil.NoSuchProcess:
    return False

"""
kill all child processes including self

return lists of pids that succeeded and failed to kill.
"""

def kill_all_child_processes(pid: Optional[int]=None) -> Tuple[List[int], List[int]]:

  if pid is None:
    pid = os.getpid()
  try:
    parent = psutil.Process(pid)
  except psutil.NoSuchProcess:
    return [], [pid]

  success = []
  failure = []
  children = list(parent.children(recursive=True)) + [parent]
  for child in children:
    if mps_safe_kill(child.pid):
      success.append(child.pid)
    else:
      failure.append(child.pid)
  return success, failure

"""
kill all process in the provided list

if recursive, kill all children as well

return lists of pids that succeeded and failed to kill.
"""
      
def kill_processes(pids: List[int], recursive=False) -> Tuple[List[int], List[int]]:
  success = []
  failure = []
  for pid in pids:
    if recursive:
      s, f = kill_all_child_processes(pid)
      success.extend(s)
      failure.extend(f)
    else:
      if mps_safe_kill(pid):
        success.append(pid)
      else:
        failure.append(pid)
  return success, failure

def is_ib_enabled() -> bool:
  return not (os.getenv("NCCL_IB_DISABLE", "False") in ["1", "true"])

def maybe_batch_isend_irecv(p2p_ops: List[dist.P2POp]):
  if not p2p_ops:
    return []
  # return dist.batch_isend_irecv(p2p_ops)

  ib_enabled = is_ib_enabled()
  # not sure why batch isend irecv hangs on finalizing collective primitives when IB is enabled; as a workaround, try to use without batching.
  # using non-batching p2p is more efficient than disabling IB completely, and the performance loss is marginal
  # confirmed: related to NCCL_BUFFSIZE, and likely a bug.

  ib_enabled = True # always use non-batching method. there seems to be bugs in nccl not limited to ib.,
  if ib_enabled:
    reqs = []
    for op in p2p_ops:
      if op.op == dist.isend:
        reqs.append(
          dist.isend(tensor=op.tensor, dst=op.peer, group=op.group)
        )
      else:
        reqs.append(
          dist.irecv(tensor=op.tensor, src=op.peer, group=op.group)
        )
    return reqs
  else:
    return dist.batch_isend_irecv(p2p_ops)
  
"""
Migration Related Memory Management
"""
import torch.multiprocessing as mp
from typing import Dict, List
import gc

class TensorPlaceholder():
  pass

def cuda_gc():
  gc.collect()
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.ipc_collect()

def forcefully_free_tensor(tensor):
  if isinstance(tensor, torch.Tensor):
    if tensor.untyped_storage().resizable():
      tensor.untyped_storage().resize_(0)
    else:
      tensor.data = torch.empty(0, device="meta")

# purge all cuda memory in a given state_dict, used for migration_save to release memory  
def forcefully_free_tensor_in_state_dict(state_dict):

  freed_bytes = 0
  def _recursive_free(value):
    nonlocal freed_bytes
    if isinstance(value, torch.Tensor):
      freed_bytes += value.numel() * value.element_size()
      forcefully_free_tensor(value)
    elif isinstance(value, Dict):
      for _, v in value.items():
        _recursive_free(v)
    elif isinstance(value, List):
      for v in value:
        _recursive_free(v)
  
  _recursive_free(state_dict)
  cuda_gc()
  return freed_bytes

def clone_and_free_tensor_in_state_dict(state_dict):
  # to clone IPC tensor before sending to the next
  # keep memory low by inplace replacement

  cloned_bytes = 0
  storage_ref_cnt = {}
  def _recursive_storage_ref_cnt(value):
    if not isinstance(value, (List, Dict)):
      return
    nonlocal storage_ref_cnt
    iterable = list(value.items()) if isinstance(value, Dict) else list(enumerate(value))
    for k, v in iterable:
      if isinstance(v, torch.Tensor):
        storage_key = v.untyped_storage().data_ptr()
        if storage_key not in storage_ref_cnt:
          storage_ref_cnt[storage_key] = 0
        storage_ref_cnt[storage_key] += 1
      else:
        _recursive_storage_ref_cnt(v)

  def _recursive_clone(value):
    if not isinstance(value, (List, Dict)):
      return
    nonlocal cloned_bytes
    iterable = list(value.items()) if isinstance(value, Dict) else list(enumerate(value))
    for k, v in iterable:
      if isinstance(v, torch.Tensor):
        nv = v.clone()
        value[k] = nv

        cloned_bytes += v.numel() * v.element_size()
        storage_key = v.untyped_storage().data_ptr()
        storage_ref_cnt[storage_key] -= 1
        if storage_ref_cnt[storage_key] == 0:
          forcefully_free_tensor(v)
      else:
        _recursive_clone(v)
        
  _recursive_storage_ref_cnt(state_dict)
  _recursive_clone(state_dict)

  return cloned_bytes
    
def interactive_IPC_state_dict_send(IPC_queues: Tuple[mp.Queue, mp.Queue], state_dict: Dict, gc_size_MB: int =1024):
  # step 1: get state dict template

  def get_state_dict_template(value):
    if isinstance(value, Dict):
      return {
        k: get_state_dict_template(v) for k, v in value.items()
      }
    if isinstance(value, List):
      return [
        get_state_dict_template(v) for v in value
      ]
    if isinstance(value, torch.Tensor) and value.device.type == "cuda":
      return TensorPlaceholder()
    return value
  
  # step 2: send the state dict template
  IPC_queues[0].put(get_state_dict_template(state_dict))

  # step 3: interactively send tensor IPC (build storage_ref_cnt, garbage collect when gc_size_MB is reached)
  storage_ref_cnt = {}
  def _recursive_storage_ref_cnt(value):
    nonlocal storage_ref_cnt
    if not isinstance(value, (List, Dict)):
      return
    iterable = list(value.items()) if isinstance(value, Dict) else list(enumerate(value))
    for k, v in iterable:
      if isinstance(v, torch.Tensor) and v.device.type == "cuda":
        storage_key = v.untyped_storage().data_ptr()
        if storage_key not in storage_ref_cnt:
          storage_ref_cnt[storage_key] = 0
        storage_ref_cnt[storage_key] += 1
      else:
        _recursive_storage_ref_cnt(v)

  _recursive_storage_ref_cnt(state_dict)

  sent_bytes = 0
  def _recursive_send(value):
    nonlocal sent_bytes
    if not isinstance(value, (Dict, List)):
      return
    iterable = list(value.items()) if isinstance(value, Dict) else list(enumerate(value))
    for k, v in iterable:
      if isinstance(v, torch.Tensor) and v.device.type == "cuda":
        IPC_queues[0].put(v, block=True)
        IPC_queues[1].get(block=True)

        storage_key = v.untyped_storage().data_ptr()        
        storage_ref_cnt[storage_key] -= 1
        if storage_ref_cnt[storage_key] == 0:
          forcefully_free_tensor(v)

        sent_bytes += v.numel() * v.element_size()
        if sent_bytes >= gc_size_MB * 1024 * 1024:
          cuda_gc()
          sent_bytes = 0
      else:
        _recursive_send(v)
    
  _recursive_send(state_dict)
  cuda_gc()


def interactive_IPC_state_dict_recv(IPC_queues: Tuple[mp.Queue, mp.Queue]) -> Dict:
  state_dict = IPC_queues[0].get(block=True)

  def _recursive_recv(value):
    if not isinstance(value, (Dict, List)):
      return
    iterable = list(value.items()) if isinstance(value, Dict) else list(enumerate(value))
    for k, v in iterable:
      if isinstance(v, TensorPlaceholder):
        tensor = IPC_queues[0].get(block = True)
        value[k] = tensor.clone()
        del tensor
        IPC_queues[1].put(None, block = True)
      else:
        _recursive_recv(v)
    
  _recursive_recv(state_dict)
  return state_dict

def instantiate_non_persistent_buffers(model: Module, device: torch.device):
  for name, buffer in list(model.named_buffers(recurse=False)):
    if buffer is not None and buffer.device != device:
      model._buffers[name] = torch.empty_like(buffer, device=device)
    
  for child in model.children():
    instantiate_non_persistent_buffers(child, device)

"""
Init all submodules parameters
"""
def reset_parameters(model: Module):
  for submodule in model.children():
    reset_parameters(submodule)
  if hasattr(model, "reset_parameters"):
    model.reset_parameters()
  else:
    for param in model.parameters(recurse=False):
      torch.nn.init.normal_(param)
