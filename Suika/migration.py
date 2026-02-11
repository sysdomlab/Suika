# we assume each paramter is identifiable by its name in entire pipeline. (that is, the param name somewhat encodes the layer_index)
from collections import OrderedDict, defaultdict
import copy
from typing import Dict, List, Any, Optional, Set, Union, Callable, Tuple
import os
from itertools import chain

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
import torch.distributed as dist
from torch.distributed import P2POp
import torch.multiprocessing as mp

from Suika.profiler import Profiler, DummyProfiler
from Suika.utils import broadcast_objects, cuda_gc, interactive_IPC_state_dict_send, interactive_IPC_state_dict_recv, maybe_batch_isend_irecv, forcefully_free_tensor, is_ib_enabled

"""
TPMeta
  - describe how tensor is sliced in TP
  - not the same as TPMeta in tensor_parallel.py, which encodes how a linear should be converted to ColumnParallelLinear and RowParallelLinear
"""
TPMeta = Tuple[int, int, int] # slice_dim, tp_index, tp_deg

"""
TensorMeta : encode basic tensor meta information
"""
class TensorMeta():
  name: Optional[str]
  dtype: torch.dtype
  shape: torch.Size
    
  tp_meta: Optional[TPMeta] # slice_dim, tp_index, tp_deg
  
  # deduce meta info from tensor. if dtype or shape provided, override original.
  def __init__(self, tensor: Optional[Tensor] = None, name: Optional[str] = None, dtype: Optional[torch.dtype] = None, shape: Optional[torch.Size] = None, tp_meta: Optional[TPMeta] = None):
    if tensor is not None:
      self.dtype = tensor.dtype
  
      if hasattr(tensor, "slice_dim"): # parallel linear
        shape = list(tensor.shape)
        shape[tensor.slice_dim] *= tensor.tp_deg
        self.shape = torch.Size(shape)
        self.tp_meta = (tensor.slice_dim, tensor.tp_index, tensor.tp_deg)
      else:
        self.shape = tensor.shape
        self.tp_meta = None
    else:
      assert dtype is not None and shape is not None, "Either tensor or (dtype, shape) should be provided"
      
    self.name = name
    
    if dtype is not None:  
      self.dtype = dtype if isinstance(dtype, torch.dtype) else torch.dtype(dtype)
    if shape is not None:
      self.shape = torch.Size(shape)
    if tp_meta is not None:
      self.tp_meta = tp_meta
      
  # return number of elements in tensor
  def numel(self) -> int:
    return torch.prod(torch.tensor(self.shape)).item() // self.tp_deg()
  
  # actual volumn required in memory / transmission in bytes
  def volumn(self) -> float:
    return self.numel() * self.dtype.itemsize
  
  # tp meta info
  def is_tp(self) -> int:
    return self.tp_meta is not None
  
  def slice_dim(self) -> int:
    if self.is_tp():
      return self.tp_meta[0]
    else:
      return None
  
  def tp_index(self) -> int:
    if self.is_tp():
      return self.tp_meta[1]
    else:
      return 0
  
  def tp_deg(self) -> int:
    if self.is_tp():
      return self.tp_meta[2]
    else:
      return 1
    
  # actual shape (tp considered)
  def actual_shape(self) -> torch.Size:
    if self.is_tp():
      shape = list(self.shape)
      shape[self.slice_dim()] //= self.tp_deg()
      return torch.Size(shape)
    else:
      return self.shape

  # rescale TensorMeta to target tp degree
  # return a list of TensorMeta, each of which is a slice of the original tensor
  # non-op if not tp, or target_tp_deg <= self.tp_deg
  def scale_to_tp(self, target_tp_deg: int) -> List["TensorMeta"]:
    if self.is_tp() and target_tp_deg > self.tp_deg():
      # tp meta info
      slice_dim, tp_index, _ = self.tp_meta
      scale = target_tp_deg // self.tp_deg()
      # assert scale > 0, f"scale_to_tp: target_tp_deg {target_tp_deg} should be larger than current tp_deg {self.tp_deg()}"
      
      return [
        TensorMeta(
                  name = self.name,
                  dtype = self.dtype,
                  shape = self.shape,
                  tp_meta = (slice_dim, scale * tp_index + i, target_tp_deg)) 
        for i in range(scale)]
    else:
      return [self]
  
  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, TensorMeta):
      return False
    tp_meta_same = self.is_tp() == other.is_tp() and self.slice_dim() == other.slice_dim()
    return tp_meta_same and self.dtype == other.dtype and self.shape == other.shape
  
  # def __getattribute__(self, name):
  #   return getattr(self, name)
  
  def __str__(self):
    return f"TensorMeta(name={self.name}, {self.dtype}, {self.shape}, {self.tp_meta})"
  def __repr__(self):
    return str(self)


def get_named_parameters_and_persistent_buffers(model: Module):
  state_dict = model.state_dict()
  for name, param in chain(model.named_parameters(), model.named_buffers()):
    if name in state_dict:
      yield name, param
    else:
      continue

"""
Extract param_meta from model, return dict[name -> param's tensor meta]
Tensor parellel is properly handled by this function
"""
def get_param_tensor_meta(model: Module) -> Dict[str, TensorMeta]:
  param_meta = {}
  for name, param in get_named_parameters_and_persistent_buffers(model): # do not use state dict, which discard param attr
      param_meta[name] = TensorMeta(tensor = param, name = name)
  return param_meta

"""
ParamLayout:
  hold information of:
    - rank -> owned param name
    - (all ranks') param name -> meta
    
  note this class is decoupled from dist group, and can be preserved across dist group reconfiguration. (though rank should be preserved across reconfig, this is accountability of external control)
"""
class ParamLayout():
  layout: Dict[int, Dict[str, TensorMeta]] # rank -> param name -> meta
  # param_meta: Dict[str, TensorMeta] 
  ## param name -> meta / assume no conflicts between ranks (may duplicate) 
  ## adding TPMeta breaks the possibility for unified param_meta

  def __init__(self):
    self.layout = {}
    # self.param_meta = {}
  

  def gather(self, self_param_meta_or_module: Union[Dict[str, TensorMeta], Module], world: Optional[dist.ProcessGroup] = None):
    
    if isinstance(self_param_meta_or_module, Module):
      self_param_meta = get_param_tensor_meta(self_param_meta_or_module)
    else:
      self_param_meta = self_param_meta_or_module
    
    param_meta_list = [None] * dist.get_world_size(world)
    
    dist.all_gather_object(param_meta_list, self_param_meta, group=world)
    for rank, param_meta in enumerate(param_meta_list):
      
      self.layout[rank] = param_meta # remebmer to ensure consistency across ranks
      
      # if check_conflict:
      #   # check no conflict
      #   for param in Intersection(self.param_meta.keys(), param_meta.keys()):
      #     if param in self.param_meta:
      #       assert self.param_meta[param] == param_meta[param], f"tensor meta conflict between {self.param_meta[param]} & {param_meta[param]}"
      
      # self.param_meta.update(param_meta)
  
  # whether param is owned by given rank
  def is_owned(self, rank: int, param: Union[str, TensorMeta]) -> bool:
    if isinstance(param, str):
      return param in self.layout[rank]
    else:
      param_name = param.name
      if param_name in self.layout[rank] and \
          self.is_tensor_tp_owned(meta_src=self.layout[rank][param_name], meta_tgt=param):
        return True
      else:
        return False
  
  # a list of owners of given param (consider tp if TensorMeta is passed in)
  def owners(self, param: Union[str, TensorMeta]) -> List[int]:
    return [rank for rank, _ in self.layout.items() if self.is_owned(rank, param)]
      
  def __str__(self):
    return f"ParamLayout(\n  {self.layout}\n)"
  
  def __repr__(self):
    return str(self)
  
  """
  utility function, given two tensor meta of same param on src dst, check if they overlap 
  (assume two meta related to same underlying param, only check tp information)
  """
  @staticmethod
  def is_tensor_tp_overlap(meta1: Union[TensorMeta, TPMeta], meta2: Union[TensorMeta, TPMeta]) -> bool:
    if isinstance(meta1, TensorMeta):
      meta1 = meta1.tp_meta
    if isinstance(meta2, TensorMeta):
      meta2 = meta2.tp_meta
    
    # at least one of them not TP
    if meta1 is None or meta2 is None:
      return True

    # both TP
    slice_dim1, tp_index1, tp_deg1 = meta1
    slice_dim2, tp_index2, tp_deg2 = meta2
    assert slice_dim1 == slice_dim2, f"slice dim mismatch {slice_dim1} vs {slice_dim2}"
    
    if tp_deg1 > tp_deg2:
      tp_index1, tp_index2 = tp_index2, tp_index1
      tp_deg1, tp_deg2 = tp_deg2, tp_deg1
      
    # assume tp_deg2 >= tp_deg1
    scale = tp_deg2 // tp_deg1

    return tp_index2 // scale == tp_index1
    # return tp_index1 * scale <= tp_index2 and tp_index2 < (tp_index1 + 1) * scale

  @staticmethod
  def is_tensor_tp_owned(meta_src: Union[TensorMeta, TPMeta], meta_tgt: Union[TensorMeta, TPMeta]) -> bool:
    if isinstance(meta_src, TensorMeta):
      meta_src = meta_src.tp_meta
    if isinstance(meta_tgt, TensorMeta):
      meta_tgt = meta_tgt.tp_meta
    src_tp_deg = meta_src[2] if meta_src is not None else 1
    tgt_tp_deg = meta_tgt[2] if meta_tgt is not None else 1
    return src_tp_deg <= tgt_tp_deg and ParamLayout.is_tensor_tp_overlap(meta_src, meta_tgt)
  

class TransferOp():
  param_name: str
  src_meta: TensorMeta
  dst_meta: TensorMeta
  src_rank: int
  dst_rank: int

  # if dst_meta not provided, assume it is the same as src_meta
  def __init__(self, param_name: str, src_rank: int, dst_rank: int, src_meta: Union[TensorMeta, Tensor], dst_meta: Optional[Union[TensorMeta, Tensor]] = None):
    self.param_name = param_name
    
    self.src_meta = src_meta if isinstance(src_meta, TensorMeta) else TensorMeta(tensor = src_meta)
    if dst_meta is None:
      self.dst_meta = self.src_meta
    else:
      self.dst_meta = dst_meta if isinstance(dst_meta, TensorMeta) else TensorMeta(tensor = dst_meta)
    
    self.src_rank = src_rank
    self.dst_rank = dst_rank
    
  
  def __str__(self):
    return f"TransferOp({self.param_name}, {self.tensor_meta}, src={self.src_rank}, dst={self.dst_rank})"
   
  def __repr__(self):
    return str(self)
  
  # return an new tensor for receiving (tp considered) allocated on given device
  def get_recv_tensor(self, device: torch.device) -> Tensor:
    shape = self.dst_meta.actual_shape()
    return torch.empty(shape, dtype=self.dst_meta.dtype, device=device)
  
  # return an contiguous tensor for sending (tp considered) chunked from given tensor
  def get_send_tensor(self, src_tensor: Tensor) -> Tensor:
    
    # assert shape and slice_dim consistency
    assert src_tensor.shape == self.src_meta.actual_shape(), f"get_send_tensor: param {self.param_name} - src_tensor shape {src_tensor.shape} mismatch with src_meta {self.src_meta.actual_shape()}"
    assert self.src_meta.slice_dim() == self.dst_meta.slice_dim(), f"get_send_tensor: param {self.param_name} - slice dim mismatch btw src {self.src_meta.slice_dim()} and dst {self.dst_meta.slice_dim()}"
    
    # case 1, both src dst not tp; 
    #         src not tp, dst tp
    #         or src tp >= dst tp (src more fine-grined, send entire tensor)
    if self.src_meta.tp_deg() >= self.dst_meta.tp_deg():
      return src_tensor
    
    # case 2, 
    #         src tp < dst tp
    (slice_dim, src_tp_index, src_tp_deg) = self.src_meta.tp_meta
    (_, dst_tp_index, dst_tp_deg) = self.dst_meta.tp_meta
    
    chunk_num = dst_tp_deg // src_tp_deg
    chunk_idx = dst_tp_index - src_tp_index * chunk_num # in a sense remainder, the chunk_idx to send
    
    send_tensor = src_tensor.chunk(chunks = chunk_num, dim = slice_dim)[chunk_idx].contiguous()
    assert send_tensor.shape == self.dst_meta.actual_shape(), f"get_send_tensor: param {self.param_name} - send_tensor shape {send_tensor.shape} mismatch with dst_meta {self.dst_meta.actual_shape()}"
    return send_tensor
      
"""
schedule_param_transfer:
  args:
    source_param_layout
    target_param_layout
  return:
    a list of computed transfer operations
"""
def schedule_param_transfer(source_param_layout: ParamLayout, target_param_layout: ParamLayout, log_func: Optional[Callable] = None) -> List[TransferOp]:
  if log_func is None:
    log_func = lambda msg: None
  
  spl, tpl = source_param_layout, target_param_layout
  # for target, parameteres that need transimission are fixed, no schedule chance
  # for source, we balance the latency of transmission from source with greedy algorithm
  #             
  # but to avoid moving from far end of the cluster while leave neighbor under utilized, we examine param transfer with fastest link first. 
  #   (fix param, let transfer rate be the fastest possible for that, then sort param according to that. after sorting, schedule each param with greedy algorithm that minimize max latency)

  # update: we observe that p2p establishing connection dominates the latency. so we change to a algorithm heuristically reduce number of connections as primary goal.

  max_tp_degree = max([param.tp_deg() for params in spl.layout.values() for param in params.values()])
  
  schedule = []
  
  latency_vec = [0.0 for rank in spl.layout.keys()]
  minimum_est_peers = 1 if is_ib_enabled() else 2 # hyperparameter to balance "minimize est peers" and "minimize max volume"

  def latency(param: TensorMeta):
    return param.volumn()
  
  for dst, dist_params in tpl.layout.items():
    est_peers = set() # estiablished peers
    for name, orig_dst_param in dist_params.items(): # enumerate required param in target
      # name: str, param: TensorMeta
      
      # rescale param to max source tp degree to support shrink in tp_degree
      tp_scaled_params = orig_dst_param.scale_to_tp(max_tp_degree)
      
      for dst_param in tp_scaled_params:
        srcs = spl.owners(dst_param)
        
        assert len(srcs) > 0, f"illegal transfer, target param {dst_param} is not present in any source - \nspl:{spl}\ntpl:{tpl}"
        
        if dst in srcs: # self owned, force src to self
          best_src = dst
          # note that we cannot set src_meta = dst_meta, since tp_meta (esp. tp_deg) may differ btw deployment, which results in tranfer failure
        else:
          # choose the best src with smallest latency (try established srcs first)
          est_srcs = [src for src in srcs if src in est_peers]
          if len(est_srcs) > 0 and len(est_peers) >= minimum_est_peers:
            best_src = min(est_srcs, key=lambda scr: latency_vec[scr] + latency(dst_param))
          else:
            best_src = min(srcs, key=lambda scr: latency_vec[scr] + latency(dst_param))

        if best_src != dst:
          log_func(f"schedule_param_transfer: {best_src} > {dst} transfer {name}")
          
        src_param = spl.layout[best_src][name]
        est_peers.add(best_src)
        
        schedule.append(
          TransferOp(param_name=name, src_rank=best_src, dst_rank=dst, src_meta=src_param, dst_meta=dst_param)
        )
        
        latency_vec[best_src] += latency(dst_param)
  
  return schedule

"""
execute a param transfer schedule
args:
  transfer_ops: a list of TransferOp
  source_state_dict: state_dict owned by this rank
  device: device to perform transfer, (ideally the same as tensors in state_dict)
  world: the process group to use for transfer, default to None
  
  note that state_dict tensor device should match the backend of world
  note that source_state_dict will be cleared in place, and becomes invalid after the call (in order to free memory)
  
return:
  target_state_dict : complete state_dict required by this rank. 
"""
def execute_param_transfer(
  transfer_ops: List[TransferOp], 
  source_state_dict: Dict[str, Dict[str, Tensor]], 
  device: torch.device, 
  expected_sub_fields: List[Tuple[str, Union[str, TensorMeta]]], 
  world: Optional[dist.ProcessGroup] = None, 
  log_func: Optional[Callable] = None,
  prof: Optional[Profiler] = None,
  forcefully_free: bool = False,
  fake_source_state_dicts: Optional[Dict] = None, # only used with checkpoint_load.
  )-> Dict[str, Dict[str, Tensor]]:
  if log_func is None:
    log_func = lambda msg: None
  if prof is None:
    prof = DummyProfiler()
  if not isinstance(device, torch.device):
    device = torch.device(device)
    
  rank = dist.get_rank(world)
  # world_size = dist.get_world_size(world)
  
  p2p_ops: List[dist.P2POp] = []
  self_transfer_ops: List[Callable] = []
  target_state_dict = {}
  slice_dim_dict = {}
  
  def is_meta_tensor(field_type: Tuple[str, Union[str, TensorMeta]]):
    return not (isinstance(field_type, str) and field_type == "normal")
  
  for op in transfer_ops:
    src, dst = op.src_rank, op.dst_rank
    param_name = op.param_name
    
    # create target state dict if not exist
    if dst == rank and not param_name in target_state_dict:
      target_state_dict[param_name] = defaultdict(list)
      slice_dim_dict[param_name] = op.dst_meta.slice_dim()
    
    if src == rank:
      assert param_name in source_state_dict, f"[{rank}]: param missing in source {param_name}"
      
      # send tensor to dst
      for (sub_field, field_type), (sub_field_backup, tensor) in zip(expected_sub_fields, source_state_dict[param_name].items()):
        # note: source_state_dict[param_name].items() may have len = 1 for buffer. zip() handles this correctly
        assert sub_field == sub_field_backup, f"expected_sub_fields mismatch {sub_field} vs {sub_field_backup}"
        
        if not is_meta_tensor(field_type):
          send_tensor = op.get_send_tensor(tensor).to(device)
        else:
          send_tensor = tensor.to(device)
          
        if src == dst:
          # self transfer
          
          ## ? not sure how cross-ref is implemented. everything seems fine even without clone.
          ## to follow the suggested best-practice, we choose to clone for safety.
          ## also, if multiple reconfiguration happens, clone is necessary
          def self_transfer_op(ctx = (param_name, sub_field, send_tensor)):
            param_name, sub_field, send_tensor = ctx
            del ctx
            target_state_dict[param_name][sub_field].append(send_tensor)

          self_transfer_ops.append(self_transfer_op)
          
        else:
          # remote transfer 
          if fake_source_state_dicts is not None: # checkpoint resume
            pass
          else: # migration
            p2p_op = P2POp(op = dist.isend, tensor = send_tensor, peer=dst, group=world)
            
            p2p_ops.append(p2p_op)
          
          # log_func(f"execute_param_transfer: {src} > {dst} send {sub_field} {TensorMeta(send_tensor, name = param_name)}")
      
    elif dst == rank:
      
      # receive tensor from src
      for sub_field, field_type in expected_sub_fields:
        
        if fake_source_state_dicts is not None: # checkpoint resume
          tensor_from_fake_source = fake_source_state_dicts[op.src_rank][param_name][sub_field]
          if not is_meta_tensor(field_type):
            recv_tensor = op.get_send_tensor(tensor_from_fake_source).to(device)
          else:
            recv_tensor = tensor_from_fake_source.to(device)
        else: # migration
          if not is_meta_tensor(field_type):
            recv_tensor = op.get_recv_tensor(device)
          else: 
            recv_tensor = torch.tensor([0], dtype=field_type.dtype, device=device)

          p2p_op = P2POp(op = dist.irecv, tensor = recv_tensor, peer=op.src_rank, group=world)
          p2p_ops.append(p2p_op)
          
        target_state_dict[param_name][sub_field].append(recv_tensor)
      
        # log_func(f"execute_param_transfer: {src} > {dst} recv {sub_field} {TensorMeta(recv_tensor, name = param_name)}")

  # overlapped clone and communicaton
  prof.start(f"execute issue p2p transfer, len {len(p2p_ops)}, numel {sum([op.tensor.numel() for op in p2p_ops])}")
  reqs = []
  log_func(f"=== issue p2p {len(p2p_ops)} ops // {torch.cuda.memory_allocated() / 1e6:.1f} / {torch.cuda.memory_reserved() / 1e6:.1f} MB ===")
  if len(p2p_ops) > 0:
    reqs.extend(maybe_batch_isend_irecv(p2p_ops))
  log_func("=== issue p2p done ===")
  prof.stop(f"execute issue p2p transfer, len {len(p2p_ops)}, numel {sum([op.tensor.numel() for op in p2p_ops])}")
  
  prof.start("execute self transfer")
  log_func(f"=== self transfer {torch.cuda.memory_allocated() / 1e6:.1f} / {torch.cuda.memory_reserved() / 1e6:.1f} MB ===")

  for op in self_transfer_ops:
    op()
  log_func(f"=== self transfer done {torch.cuda.memory_allocated() / 1e6:.1f} / {torch.cuda.memory_reserved() / 1e6:.1f} MB ===")

  prof.stop("execute self transfer")

  log_func("=== wait for p2p ===")
  prof.start("execute wait p2p transfer")
  for req in reqs:
    req.wait()
  prof.stop("execute wait p2p transfer")
  log_func("=== wait for p2p done ===")
          
  # merge the list of tensors into a single tensor
  prof.start("execute post process")

  del p2p_ops
  del self_transfer_ops
  del reqs
  source_state_dict.clear() # a clear is required, since the caller of execute_param_transfer is still holding a reference to it.
  del source_state_dict
  cuda_gc()
  log_func(f"=== tail process {torch.cuda.memory_allocated() / 1e6:.1f} / {torch.cuda.memory_reserved() / 1e6:.1f} MB ===")
  for param_name, sub_fields in target_state_dict.items():
    for (sub_field, field_type), (sub_field_backup, tensors) in zip(expected_sub_fields, sub_fields.items()):
      
      if not is_meta_tensor(field_type) and len(tensors) > 1:
        # log_func(f"concat {param_name}-{tensors}-{slice_dim_dict[param_name]}")
        target_state_dict[param_name][sub_field] = torch.cat(tensors, dim=slice_dim_dict[param_name])
      else:
        target_state_dict[param_name][sub_field] = tensors[0].clone() # avoid viewing into a larger tensor, and hence pin unnecessary memory
      if forcefully_free:
        for tensor in tensors:
          forcefully_free_tensor(tensor)
      del tensors
    del sub_fields
  cuda_gc()
  log_func(f"=== tail process done {torch.cuda.memory_allocated() / 1e6:.1f} / {torch.cuda.memory_reserved() / 1e6:.1f} MB ===")
    
  prof.stop("execute post process")


  # there's a bug with device_id arg in init_process_group with pytorch 2.6.0, that collective ops may seg_fault after p2p ops if device_id is set
  # A barrier here will hackily address the issue, 
  # but it is in general recommended not to use device_id arg, or wait till the bug is fixed
  
  # dist.barrier(group=world)
  
  return target_state_dict

def schedule_and_execute_param_transfer(
  source_param_layout: ParamLayout, 
  target_param_layout: ParamLayout, 
  source_state_dict: Dict[str, Dict[str, Tensor]],
  other_context_dict: Dict[str, Any],
  device: torch.device,
  world: Optional[dist.ProcessGroup],
  prof: Profiler,
  forcefully_free: bool,
  log_func=None
  ):
  if not isinstance(device, torch.device):
    device = torch.device(device)

  # compute tranfser schedule
  prof.start("schedule")
  schedule = schedule_param_transfer(
    source_param_layout=source_param_layout, 
    target_param_layout=target_param_layout, 
    # log_func=log_func
  )
  prof.stop("schedule")
  
  prof.start("execute", require_memory=True)
  # execute transfer schedule
  # the on-load to GPU will be handled here
  target_state_dict = execute_param_transfer(
    transfer_ops=schedule, 
    source_state_dict=source_state_dict,
    device = device,
    expected_sub_fields=other_context_dict["expected_sub_fields"],
    world = world,
    forcefully_free=forcefully_free,
    prof = prof,
    log_func=log_func
  )
  if device.type == "cuda":
    torch.cuda.synchronize(device=device)
  prof.stop("execute", require_memory=True)
  
  return target_state_dict



## Deprecated
# """
# get state_dict from model, perform all_gather for parallel parameters if necessary

# the function is discarded after tp shareded tranfser is implemented, but if you need traditional complete checkpoint-resume, you may use it as utility
# """
# def get_gathered_state_dict(model: Module, device: torch.device, log_func: Optional[Callable] = None) -> Dict[str, Tensor]: 
#   if log_func is None:
#     log_func = lambda msg: None
#   gathered = {}
  
#   post_hooks = []
  
#   for name, param in model.named_parameters(): # do not use state_dict, which discard param's attribute
#     if hasattr(param, "slice_dim"):
#       log_func(f"state dict gather: {param}")
         
#       gather_list = [torch.empty_like(param) for _ in range(param.tp_deg)]
      
#       req = dist.all_gather(gather_list, param.contiguous(), group=param.tp_group, async_op = True)
      
#       def restore(ctx = (req, name, param, gather_list, device)):
#         req, name, param, gather_list, device = ctx
#         req.wait()
#         gathered[name] = torch.cat(gather_list, dim=param.slice_dim).to(device)
      
#       post_hooks.append(restore)      
#     else:
#       gathered[name] = param.to(device)

#   for hook in post_hooks:
#     hook()
    
#   return gathered

# """
# prepare state_dict for model, chunk if necessary for parallel linear


# the function is discarded after tp shareded tranfser is implemented, but if you need traditional complete checkpoint-resume, you may use it as utility
# """
# def get_partitioned_state_dict(model: Module, state_dict: Dict[str, Tensor], log_func: Optional[Callable] = None) -> Dict[str, Tensor]:
#   # assume state_dict are placed on proper device
#   # state_dict will become invalid after this function
#   if log_func is None:
#     log_func = lambda msg: None
  
#   partitioned = {}
#   for name, param in model.named_parameters():
#     assert name in state_dict, f"missing {name} in state_dict"

#     if hasattr(param, "slice_dim"):
#       log_func(f"state dict partitioned: {param}")
      
#       # split the tensor along the slice_dim
#       partitioned[name] = state_dict[name].chunk(chunks = param.tp_deg, dim=param.slice_dim)[param.tp_index].clone()
      
#       state_dict[name] = None # free tensor
#     else:
#       partitioned[name] = state_dict[name]
#   return partitioned

"""
Merge model and optimizer state dict, and other context
  1. merge param-specific optimizer state into model_state_dict
  2. merge optimizer param_groups into other_context

return 
  1. merged state dict 
  2. other_context with params_group as a Dict

"""

def merge_model_and_optimizer_state_dict(model: Module, optimizer: Optional[Optimizer], other_context: Optional[Any]) -> Tuple[Dict, Dict]:
  # merge optimizer state dict [state] to model state dict, merge [param_groups] fields to other_context
  model_state_dict = model.state_dict()

  if optimizer:  
    optimizer_state_dict = optimizer.state_dict()
    
    other_context_dict = {"other_context": other_context, "param_groups": optimizer_state_dict["param_groups"]}
    optimizer_state_dict = optimizer_state_dict["state"]
    
  else:
    optimizer_state_dict = {}
    other_context_dict = {"other_context": other_context}
  
  merged_state_dict = {}
  for idx, (name, param) in enumerate(get_named_parameters_and_persistent_buffers(model)):
    merged_entry = {
      "param": model_state_dict[name],
    }
    
    optimizer_entry = optimizer_state_dict.get(idx, {})
    merged_entry.update(
      optimizer_entry
    )
    
    merged_state_dict[name] = merged_entry
    
    if not "expected_sub_fields" in other_context_dict:
      expected_sub_fields = []
      for sub_field, tensor in merged_entry.items():
        if tensor.numel() == 1: # singleton tensor, special meta data of optimizer (e.g. lr, weight_decay)
          expected_sub_fields.append((sub_field, TensorMeta(tensor)))
        else:
          expected_sub_fields.append((sub_field, "normal"))
      other_context_dict["expected_sub_fields"] = expected_sub_fields
    
  return merged_state_dict, other_context_dict

"""
Split model and optimizer state dict, retrieve other_context.

return:
  1. model state_dict
  2. optimizer state_dict
  3. other_context
"""

def split_model_and_optimizer_state_dict(model: Module, merged_state_dict: Dict[str, Dict[str, Tensor]], other_context_dict: Dict[str, Any]) -> Tuple[Dict, Dict, Any]:
  # split optimizer state dict
  model_state_dict = {}
  name_indexed_optimizer_state_dict = {}
  
  for name, entry in merged_state_dict.items():
    model_state_dict[name] = entry["param"]
    del entry["param"]
    name_indexed_optimizer_state_dict[name] = entry
    
  # reconvert name based optimizer state dict to param id based
  param_indexed_optimizer_state_dict = {}
  for idx, (name, param) in enumerate(get_named_parameters_and_persistent_buffers(model)):
    param_indexed_optimizer_state_dict[idx] = name_indexed_optimizer_state_dict[name]
  
  # split other_context
  other_context = other_context_dict["other_context"]
  param_groups = other_context_dict.get("param_groups", None)
  
  # whether optimizer is None is not checked here
  optimizer_state_dict = {
    "state": param_indexed_optimizer_state_dict,
    "param_groups": param_groups,
  }
  return model_state_dict, optimizer_state_dict, other_context

# Deprecated
"""
Perform all necessary migration operations to save the model state.

checkpoint path should on local storage
"""
def migration_save_disk(model: Module, 
                        optimizer: Optional[Optimizer] = None,
                        other_context: Optional[Any] = None,
                        device: torch.device = "cpu", 
                        checkpoint_path = "./", 
                        file_prefix = "",
                        world: Optional[dist.ProcessGroup] = None,
                        log_func: Optional[Callable] = None, 
                        profiler: Optional[Profiler] = None):
  if log_func is None:
    log_func = lambda msg: None
  if profiler is None:
    profiler = DummyProfiler()
    
  rank = dist.get_rank(world) if dist.is_initialized() else 0
  
  state_dict_path = os.path.join(checkpoint_path, f"{file_prefix}_state_dict_{rank}.pt")
  param_layout_path = os.path.join(checkpoint_path, f"{file_prefix}_param_layout.pt")
  other_context_path = os.path.join(checkpoint_path, f"{file_prefix}_other_context.pt")
  
  # make dir
  os.makedirs(checkpoint_path, exist_ok=True)
  
  state_dict, other_context_dict = merge_model_and_optimizer_state_dict(model, optimizer, other_context)
  
  if not dist.is_initialized():
    # local training
    log_func("migration_save_disk: dist not initialized, assume local training")
    
    # save state_dict and other context
    torch.save(state_dict, state_dict_path)
    torch.save(other_context_dict, other_context_path)
    
  else:  
    # distributed training
    
    # set current device for nccl
    device = torch.device(device) if not isinstance(device, torch.device) else device
    if device.type == "cuda":
      torch.cuda.set_device(device.index)
    
    self_param_meta = get_param_tensor_meta(model)
    
    param_layout = ParamLayout()
    param_layout.gather(self_param_meta, world)

    # now we do not need gather, partitioned tensor is saved distributedly
    # state_dict = get_gathered_state_dict(model=model, device=device, log_func = log_func)

    # save param_layout and other context on rank 0 only
    if rank == 0:
      torch.save(param_layout, param_layout_path)
      torch.save(other_context_dict, other_context_path)
  
    # save state_dict
    torch.save(state_dict, state_dict_path)

    dist.barrier(group=world)
  
"""
Perfrom all necessary migration opeartion to load the model state.

Disk verison
"""
def migration_load_disk(model: Module,
                        optimizer: Optional[Optimizer] = None,
                        device: torch.device = "cpu", 
                        checkpoint_path = "./", 
                        file_prefix = "", # for shm only, do not explicitly use
                        remove_after_load: bool = False,
                        world: Optional[dist.ProcessGroup] = None, 
                        log_func: Optional[Callable] = None, 
                        profiler: Optional[Profiler] = None
                        ) -> Union[Any, None]:
  if log_func is None:
    log_func = lambda msg: None
  if profiler is None:
    profiler = DummyProfiler()
      
  rank = dist.get_rank(world) if dist.is_initialized() else 0
  
  source_state_dict_path = os.path.join(checkpoint_path, f"{file_prefix}_state_dict_{rank}.pt")
  source_param_layout_path = os.path.join(checkpoint_path, f"{file_prefix}_param_layout.pt")
  other_context_path = os.path.join(checkpoint_path, f"{file_prefix}_other_context.pt")
  
  if not dist.is_initialized():
    # local training
    target_state_dict = torch.load(source_state_dict_path, weights_only=False,
                                   mmap=checkpoint_path.startswith(sys_shm_path))
    other_context = torch.load(other_context_path, weights_only=False)
    
    if remove_after_load:
      os.remove(source_state_dict_path)
      os.remove(other_context_path)  
    
  else:
    # distributed training

    # set current device for nccl
    device = torch.device(device) if not isinstance(device, torch.device) else device
    if device.type == "cuda":
      torch.cuda.set_device(device.index)

    if rank == 0:
      try:
        source_param_layout = torch.load(source_param_layout_path, weights_only=False)
        other_context_dict = torch.load(other_context_path, weights_only=False)
        if remove_after_load:
          os.remove(source_param_layout_path)
          os.remove(other_context_path)
      except FileNotFoundError as e:
        source_param_layout = e
        other_context_dict = None
        # we cannot exit here, as ranks other than 0 awaits for broadcast
        # so we reuse source_param_layout as error in case of FileNotFoundError
    else:
      source_param_layout = None
      other_context_dict = None
    
    # broadcast source param layout / other_context
    broadcast_list = [source_param_layout, other_context_dict]
    dist.broadcast_object_list(broadcast_list, src=0, group=world)
    source_param_layout = broadcast_list[0]
    other_context_dict = broadcast_list[1]
    
    if isinstance(source_param_layout, FileNotFoundError): # file not found
      raise source_param_layout
    
    if os.path.exists(source_state_dict_path):
      source_state_dict = torch.load(source_state_dict_path, weights_only=False, 
                                     mmap=checkpoint_path.startswith(sys_shm_path)) 
      if remove_after_load:
        os.remove(source_state_dict_path)
    else:
      source_state_dict = {}
    
    target_param_layout = ParamLayout()
    target_param_layout.gather(model, world)
    
    # compute tranfser schedule
    schedule = schedule_param_transfer(
      source_param_layout=source_param_layout, 
      target_param_layout=target_param_layout, 
      log_func=log_func
      )
    
    # execute transfer schedule
    target_state_dict = execute_param_transfer(transfer_ops=schedule, 
                                              source_state_dict=source_state_dict,
                                              device = device,
                                              expected_sub_fields=other_context_dict["expected_sub_fields"],
                                              world = world,
                                              log_func=log_func
                                              )
    
    dist.barrier(group=world)
    
    # now we do not need gather, partitioned tensor is send in partitioned way
    # target_state_dict_partitioned = get_partitioned_state_dict(model, target_state_dict, log_func = log_func)
    
    # model.load_state_dict(target_state_dict_partitioned)
  
  model_state_dict, optimizer_state_dict, other_context = \
    split_model_and_optimizer_state_dict(model=model, merged_state_dict=target_state_dict, other_context_dict=other_context_dict)
  
  model_state_dict = OrderedDict(sorted(model_state_dict.items(), key=lambda x: x[0]))

  model.load_state_dict(model_state_dict, assign=True)
  if optimizer:
    optimizer.load_state_dict(optimizer_state_dict)
  
  return other_context
  
"""
Perfrom all necessary migration opeartion to load the model state.

Shared Memory verison

... kind of redundant. as we access filesystem to access shared memory anyway
# done, merged to disk version
"""
# assume linux
sys_shm_path = "/dev/shm"

def migration_save_shm(
                  model: Module, 
                  optimizer: Optional[Optimizer] = None,
                  other_context: Optional[Any] = None,
                  device: torch.device = "cpu",
                  shm_name: str = "shm",
                  world: Optional[dist.ProcessGroup] = None, 
                  log_func: Optional[Callable] = None,
                  profiler: Optional[Profiler] = None
                  ):

    return migration_save_disk(
      model=model,
      optimizer=optimizer,
      other_context=other_context,
      device=device,
      checkpoint_path=sys_shm_path,
      file_prefix=shm_name,
      world=world,
      log_func=log_func,
      profiler=profiler
    )

def migration_load_shm(
                    model: Module, 
                    optimizer: Optional[Optimizer] = None,
                    device: torch.device = "cpu",
                    shm_name: str = "shm",
                    world: Optional[dist.ProcessGroup] = None, 
                    log_func: Optional[Callable] = None,
                    profiler: Optional[Profiler] = None
                  ) -> Union[Any, None]:
    return migration_load_disk(
      model=model,
      optimizer=optimizer,
      device=device,
      checkpoint_path=sys_shm_path,
      file_prefix=shm_name,
      remove_after_load=True, # for shm transfer, must remove after load
      world=world,
      log_func=log_func,
      profiler=profiler
    )
    

"""
DEPRECATED

@EXPORT
Saves the model state for migration using either shared memory or disk storage.

Args:
    model (Module): The PyTorch model to save
    other_context (Optional[Any]): Additional context to save with the model, and will be broadcast at load (must be pickable)
    device (torch.device): Device to use for communication and storage
                          (default to "cpu")
    shm_name (Optional[str]): Name prefix for shared memory segments when using shared memory
                             (mutually exclusive with checkpoint_path)
    checkpoint_path (Optional[str]): Path to save checkpoint files when using disk storage
                                    (mutually exclusive with shm_name)
    world (Optional[dist.ProcessGroup]): The process group for distributed communication
    log_func (Optional[Callable]): Function for logging messages, defaults to no-op
    profiler (Optional[Profiler]): Profiler instance to track performance, defaults to dummy profiler

Returns:
    Result of the underlying save operation

Notes:
    - Either shm_name or checkpoint_path must be provided, but not both
    - When using shared memory, files are created in /dev/shm/
    - For TP (tensor parallel) models, state is distributed across ranks
"""

def migration_save_legacy(
                  model: Module, 
                  optimizer: Optional[Optimizer] = None,
                  other_context: Optional[Any] = None,
                  device: torch.device = "cpu",
                  shm_name: Optional[str] = None,
                  checkpoint_path: Optional[str] = None,
                  world: Optional[dist.ProcessGroup] = None, 
                  log_func: Optional[Callable] = None,
                  profiler: Optional[Profiler] = None
                  ):
  if log_func is None:
    log_func = lambda msg: None
  if profiler is None:
    profiler = DummyProfiler()
      
  log_func("migration_save: begin")
  profiler.start("migration_save", True, True, True)
  
  assert (shm_name is not None) ^ (checkpoint_path is not None), "exactly one of checkpoint_path or shm_name must be provided"
  
  # choose shared memory or disk
  if shm_name: # shared memory
    
    result = migration_save_shm(
      model=model, 
      optimizer=optimizer,
      other_context=other_context,
      device=device,
      shm_name=shm_name,
      world=world,
      log_func=log_func,
      profiler=profiler
    )
  else: # disk  
    
    result = migration_save_disk(
      model=model, 
      optimizer=optimizer,
      other_context=other_context,
      device=device,
      checkpoint_path=checkpoint_path,
      world=world,
      log_func=log_func,
      profiler=profiler
    )
    
  profiler.stop("migration_save", True, True, True)
  log_func("migration_save: end")
  
  return result

"""
DEPRECATED.

@EXPORT
Loads the model state for migration from either shared memory or disk storage.

Args:
    model (Module): The PyTorch model to load state into
    device (torch.device): Device to use for communication and storage
                          (default to "cpu")
    shm_name (Optional[str]): Name prefix for shared memory segments when loading from shared memory
                             (mutually exclusive with checkpoint_path)
    checkpoint_path (Optional[str]): Path to load checkpoint files when using disk storage
                                    (mutually exclusive with shm_name)
    remove_after_load (Optional[bool]): Whether to remove the checkpoint files after loading
                                       (only applicable for disk storage)
    not_exists_ok (Optional[bool]): If True, do not raise an error if the checkpoint file does not exist
                                    
    world (Optional[dist.ProcessGroup]): The process group for distributed communication
    log_func (Optional[Callable]): Function for logging messages, defaults to no-op
    profiler (Optional[Profiler]): Profiler instance to track performance, defaults to dummy profiler

Returns:
    other_context saved at migration_save, or None if not saved

Notes:
    - Either shm_name or checkpoint_path must be provided, but not both
    - When using shared memory, files are loaded from /dev/shm/ and removed after loading
    - Parameters are transfered in batch and in sharded pattern with minimal volumn possible
"""
def migration_load_legacy(  
                  model: Module, 
                  optimizer: Optional[Optimizer] = None,
                  device: torch.device = "cpu",
                  shm_name: Optional[str] = None,
                  checkpoint_path: Optional[str] = None,                  
                  remove_after_load: Optional[bool] = None,
                  allow_not_exists: bool = True,
                  world: Optional[dist.ProcessGroup] = None, 
                  log_func: Optional[Callable] = None,
                  profiler: Optional[Profiler] = None
                  ) -> Union[Any, None]:
  if log_func is None:
    log_func = lambda msg: None
  if profiler is None:
    profiler = DummyProfiler()
  
  log_func("migration_load: begin")
  profiler.start("migration_load", True, True, True)
    
  assert (shm_name is not None) ^ (checkpoint_path is not None), "exactly one of checkpoint_path or shm_name must be provided"
  
  try:
    # choose shared memory or disk
    if shm_name: # shared memory
      if remove_after_load is not None:
        log_func("migration_load: [warning] remove_after_load arg is not supported for shm")
        
      result = migration_load_shm(
        model=model, 
        optimizer=optimizer,
        device=device,
        shm_name=shm_name,
        world=world,
        log_func=log_func,
        profiler=profiler
      )
    else: # disk 
      
      result = migration_load_disk(
        model=model, 
        optimizer=optimizer,
        device=device,
        checkpoint_path=checkpoint_path,
        remove_after_load=remove_after_load,
        world=world,
        log_func=log_func,
        profiler=profiler
      )
  except FileNotFoundError as e:
    if allow_not_exists:
      log_func(f"migration_load: [warning] checkpoint file not found, skip loading")
      result = None
    else:
      raise e
  profiler.stop("migration_load", True, True, True)
  log_func("migration_load: end")
  
  return result

"""
Traditional Checkpoint-Resume Mechanism
"""
def checkpoint_save(model: Module, optimizer: Union[Optimizer, None], other_context: Any,checkpoint_dir: str, world: Optional[dist.ProcessGroup] = None, log_func: Optional[Callable] = None):
  log_func("checkpoint_save: begin")

  prof = Profiler(strategy="last")
  
  os.makedirs(checkpoint_dir, exist_ok=True)
  
  prof.start("param_layout")
  param_layout = ParamLayout()
  param_layout.gather(model)
  prof.stop("param_layout")

  prof.start("merge")
  state_dict, other_context_dict = merge_model_and_optimizer_state_dict(model, optimizer, other_context)
  prof.stop("merge")
  
  rank = dist.get_rank(world)
  
  prof.start("save")
  if rank == 0:
    torch.save(param_layout, os.path.join(checkpoint_dir, "param_layout.pt"))
    torch.save(other_context_dict, os.path.join(checkpoint_dir, "other_context.pt"))
  
  torch.save(state_dict, os.path.join(checkpoint_dir, f"state_dict_{rank}.pt"))
  prof.stop("save")
  
  log_func(f"checkpoint_save: end")
  log_func(f"{prof.full_report_str()}")
  return
  
def checkpoint_load(model: Module, optimizer: Union[Callable, Optimizer, None], checkpoint_dir: str, world: Optional[dist.ProcessGroup] = None, log_func: Optional[Callable] = None):
  log_func(f"checkpoint_load: begin")
  log_func(f"memory: {torch.cuda.memory_allocated() / 1e6} / {torch.cuda.memory_reserved() / 1e6} MB")

  prof = Profiler(strategy="last")

  rank = dist.get_rank(world)
  
  if isinstance(optimizer, Callable):
    optimizer_provider = optimizer
    del optimizer

  try:
    prof.start("param_layout")
    source_param_layout: ParamLayout = torch.load(os.path.join(checkpoint_dir, "param_layout.pt"), weights_only=False, map_location='cpu')
    other_context_dict = torch.load(os.path.join(checkpoint_dir, "other_context.pt"), weights_only=False, map_location='cpu')
    
    target_param_layout = ParamLayout()
    target_param_layout.gather(model)
    prof.stop("param_layout")

    prof.start("schedule")
    transfer_ops = schedule_param_transfer(
      source_param_layout=source_param_layout, 
      target_param_layout=target_param_layout
    )
    prof.stop("schedule")
    
    srcs = [op.src_rank for op in transfer_ops if op.dst_rank == rank]
    if rank in source_param_layout.layout:
      srcs += [rank] # ensure self is loaded. sometime schedule may decide no self transfer, lol, so self may be missing in srcs.
    srcs = list(set(srcs))
    
    log_func(f"checkpoint_load: srcs = {srcs}")
    prof.start("state_dict")
    log_func(f"memory: {torch.cuda.memory_allocated() / 1e6} / {torch.cuda.memory_reserved() / 1e6} MB")
    dist.barrier(world)
    fake_source_state_dicts = {
      src: torch.load(os.path.join(checkpoint_dir, f"state_dict_{src}.pt"), weights_only=False, map_location='cpu', mmap=False)
      # our remote storage does not support mmap, and will slow down disk loading by x10 if set to True
      for src in srcs
    }
    prof.stop("state_dict")
    
    log_func(f"checkpoint_load: srcs loaded")
    log_func(f"memory: {torch.cuda.memory_allocated() / 1e6} / {torch.cuda.memory_reserved() / 1e6} MB")
    self_source_state_dict = fake_source_state_dicts[rank] if rank in srcs else {}
    
    prof.start("execute")
    traget_state_dict = execute_param_transfer(
      transfer_ops=transfer_ops,
      source_state_dict=self_source_state_dict,
      device="cpu",
      expected_sub_fields=other_context_dict["expected_sub_fields"],
      world=world,
      forcefully_free=False,
      fake_source_state_dicts=fake_source_state_dicts,
      log_func=log_func
    )
    prof.stop("execute")
    del fake_source_state_dicts

    log_func(f"checkpoint_load: execute_param_transfer done")
    log_func(f"memory: {torch.cuda.memory_allocated() / 1e6} / {torch.cuda.memory_reserved() / 1e6} MB")
    
    prof.start("load")
    model_state_dict, optimizer_state_dict, other_context = \
      split_model_and_optimizer_state_dict(model, 
                                            merged_state_dict=traget_state_dict, 
                                            other_context_dict=other_context_dict)
    
    model_state_dict = OrderedDict(sorted(model_state_dict.items(), key=lambda x: x[0]))

    model.load_state_dict(model_state_dict, assign=False) # note: migration's default assign have been switched to True

    log_func(f"checkpoint_load: model loaded")
    log_func(f"memory: {torch.cuda.memory_allocated() / 1e6} / {torch.cuda.memory_reserved() / 1e6} MB")

    if optimizer_provider:
      optimizer = optimizer_provider()
    if optimizer:
      optimizer_state_dict["param_groups"][0]["params"] = optimizer.state_dict()["param_groups"][0]["params"]
      optimizer.load_state_dict(optimizer_state_dict)
    log_func(f"memory: {torch.cuda.memory_allocated() / 1e6} / {torch.cuda.memory_reserved() / 1e6} MB")

    log_func(f"checkpoint_load: optimizer loaded")
    prof.stop("load")

    log_func(f"checkpoint_load: end")
    log_func(f"{prof.full_report_str()}")

        
  except FileNotFoundError as e:
    log_func(f"checkpoint_load: [warning] checkpoint file not found, skip loading\n{e}")
    if hasattr(model, "instantiate") and getattr(model, "on_meta", False):
      model.instantiate()
      
    if optimizer_provider:
      return optimizer_provider(), None
    else:
      return None
    
  dist.barrier(world) # sync before remove
  if rank == 0:
    log_func(">> before remove")
    os.remove(os.path.join(checkpoint_dir, "param_layout.pt"))
    os.remove(os.path.join(checkpoint_dir, "other_context.pt"))
    while True:
      try:
        os.remove(os.path.join(checkpoint_dir, f"state_dict_{rank}.pt"))
      except FileNotFoundError:
        break
      rank += 1
    log_func(">> after remove")
    
  log_func(f"checkpoint_load: end")
  if optimizer_provider:
    return optimizer, other_context
  else:
    return other_context



"""
Reimplement migration_save / load, avoid using torch.save/load where unnecessary serialization dominates overhead 
"""

dummy_migration_IPC_state_dict = {
  "source_param_layout": None,
  "source_state_dict": {},
  "other_context_dict": {},
}

"""
@EXPORT

Name: 
  migration_save

Saves the model and optimizer state along with other context via Inter-Process Communication (IPC) queues.
This version aims to reduce overhead compared to disk/shm-based torch.save by directly passing
data structures (layouts, context) and potentially tensors (if queues support efficient tensor transfer)
between processes, minimizing serialization where possible.

Args:
    IPC_queues (Tuple[mp.Queue, mp.Queue]): 
      A tuple containing two multiprocessing queues:
        - IPC_queues[0] (Forward Queue): Used by this function to send the state dictionary
          to the loading process.
        - IPC_queues[1] (Backward Queue): Used by this function to wait for an acknowledgment
          signal from the loading process after it has finished.
    model (Module): 
      The PyTorch model whose state needs to be saved.
    optimizer (Optional[Optimizer]): 
      The optimizer whose state needs to be saved.
    other_context (Optional[Any]): 
      Additional arbitrary context data to save (must be picklable).
    device (torch.device, optional):
      The device context to use, primarily for setting the
        active CUDA device if applicable. Defaults to `torch.get_default_device()`.
    world (Optional[dist.ProcessGroup], optional): 
      The process group for distributed operations like gathering the parameter layout. Defaults to None.
    log_func (Optional[Callable], optional): 
      A logging function. Defaults to a no-op.
    checkpoint_resmue: Optional[str]:
      If specified, traditional checkpoint-resume mechanism will be used.
        checkpoint_resmue is the path to per-job unique checkpoint directory on NFS
      Defaults to None.
    
"""

def migration_save(
                   IPC_queues: Tuple[mp.Queue, mp.Queue],
                   model: Module,
                   optimizer: Optional[Optimizer] = None,
                   other_context: Optional[Any] = None,
                   device: torch.device = None,
                   world: Optional[dist.ProcessGroup] = None,
                   log_func: Optional[Callable] = None,
                   checkpoint_resume: Optional[str] = None,
                  ):
  if log_func is None:
    log_func = lambda msg: None
  if device is None:
    device = torch.get_default_device()
  elif not isinstance(device, torch.device):
    device = torch.device(device)
  if device.type == "cuda":
    torch.cuda.set_device(device.index)
  prof = Profiler(strategy="last", device=device)
  
  log_func(f"migration_save: begin")
  # log_func(f"current memory: {torch.cuda.memory_reserved() / 1024 ** 2: .2f}MB")
  torch.cuda.empty_cache()
  # log_func(f"after empty_cache memory: {torch.cuda.memory_reserved() / 1024 ** 2: .2f}MB")

  if checkpoint_resume is not None:
    # use traditional checkpoint-resume mechanism
    log_func(f"migration_save: checkpoint-resume={checkpoint_resume}")
    return checkpoint_save(
      model=model,
      optimizer=optimizer,
      other_context=other_context,
      checkpoint_dir=checkpoint_resume,
      world=world,
      log_func=log_func
    )
  
  
  # handshake with migration_load to determine if is_scale_up
  rank = dist.get_rank(world)
  if rank == 0:
    old_world_size = dist.get_world_size(world)
    IPC_queues[0].put(old_world_size)
    new_world_size = IPC_queues[1].get(block = True)
    is_scale_up = new_world_size >= old_world_size
  else:
    is_scale_up = None
    new_world_size = None
  is_scale_up, new_world_size = broadcast_objects(is_scale_up, new_world_size, group=world)
  
  log_func(f"migration_save: is_scale_up = {is_scale_up}")
  
  # get source_param_layout
  log_func(f"<< before gather param layout")
  prof.start("param_layout")
  param_layout = ParamLayout()
  param_layout.gather(model, world)
  prof.stop("param_layout")
  log_func(f"<< after gather param layout")
  
  # get source_state_dict
  log_func("<< before load")
  prof.start("merge", require_memory=True)
  source_state_dict, other_context_dict = merge_model_and_optimizer_state_dict(model, optimizer, other_context)
  prof.stop("merge", require_memory=True)
  log_func("<< after load")
  
  if is_scale_up:
    # scale up, simply send source to migration_load
    
    # send state_dict and other_context_dict via IPC queues
    log_func("<< before send")
    prof.start("send", require_memory=True)
    IPC_state_dict = {
      "source_param_layout": param_layout,
      "source_state_dict": source_state_dict,
      "other_context_dict": other_context_dict,
    }
    
    interactive_IPC_state_dict_send(IPC_queues=IPC_queues, state_dict=IPC_state_dict)
    del IPC_state_dict, source_state_dict, other_context_dict
    prof.stop("send", require_memory=True)
    log_func("<< after send")
    
  else:
    # not scale up, recv target param_layout (rank 0 only), broadcast, perform transfer, then send target_state_dict
    if rank == 0:
      # cannot expect shrank process to get target_param_layout from train_load
      # but rank 0 is always valid
      target_param_layout = IPC_queues[1].get(block = True)
    else:
      target_param_layout = None
    target_param_layout = broadcast_objects(target_param_layout, group=world)
    
    # param transfer
    log_func("<< before transfer")
    prof.start("transfer", require_memory=True)
    target_state_dict = schedule_and_execute_param_transfer(
      source_param_layout=param_layout,
      target_param_layout=target_param_layout,
      source_state_dict=source_state_dict,
      other_context_dict=other_context_dict,
      device=device,
      world=world,
      prof=prof,
      forcefully_free=True,
      log_func=log_func
    )
    del source_state_dict
    prof.stop("transfer", require_memory=True)
    log_func("<< after transfer")
    
    # send target_state_dict and other_context_dict via IPC queues
    log_func("<< before send")
    prof.start("send", require_memory=True)
    IPC_state_dict = {
      "target_state_dict": target_state_dict,
      "other_context_dict": other_context_dict,
    }
    has_recver = not (rank >= new_world_size)
    if has_recver:
      interactive_IPC_state_dict_send(IPC_queues=IPC_queues, state_dict=IPC_state_dict)
    del IPC_state_dict, target_state_dict, other_context_dict
    prof.stop("send", require_memory=True)
    log_func("<< after send")

    
  log_func(f"migration_save: {prof.full_report_str(require_memory=True)}")  
  log_func(f"migration_save: end")
  
  

"""
@EXPORT
Loads the model and optimizer state along with other context received via Inter-Process Communication (IPC) queues.
This complements `migration_save` by receiving the state dictionary, performing necessary distributed transfers
based on source and target layouts, and loading the state into the provided model and optimizer.

Name:
    migration_load

Args:
    IPC_queues (Tuple[mp.Queue, mp.Queue]): 
      A tuple containing two multiprocessing queues:
        - IPC_queues[0] (Forward Queue): Used by this function to receive the state dictionary
          from the saving process.
        - IPC_queues[1] (Backward Queue): Used by this function to send an acknowledgment signal
          back to the saving process once loading is complete (or determined unnecessary).
    model (Module): 
      The PyTorch model to load the state into.
    optimizer (Optional[Union[Optimizer, Callable]]): 
      The optimizer to load the state into.
      1. if None, optimizer state is skipped
      2. if Optimizer, load the state into the optimizer
      3. if Callable, it should take no input and return a new optimizer
          in this case, the optimizer is returned from the function
        (this is to avoid instantiating model to create optimizer before load, to save GPU memory)
        
        e.g. 
          def optimizer_provider(state_dict):
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            return optimizer
        the model is guaranteeded to be instantiated if on meta before calling this function
      
    device (torch.device, optional): 
      The target device for loaded tensors and distributed communication.
      Defaults to `torch.get_default_device()`.
    world (Optional[dist.ProcessGroup], optional): 
      The process group for distributed operations
      like broadcasting metadata and executing parameter transfers. Defaults to None.
    log_func (Optional[Callable], optional): 
      A logging function. Defaults to a no-op.
    checkpoint_resmue: Optional[str]:
      If specified, traditional checkpoint-resume mechanism will be used.
      checkpoint_resmue is the path to per-job unique checkpoint directory on NFS
      Defaults to None.

Returns:
    Union[Any, None]: 
    1. If optimizer is not Callable: 
        The `other_context` that was saved, or None if no valid checkpoint
                      was received (e.g., on a fresh process launch).
    Tuple[Optimizer, Union[Any, None]]
    2. If optimizer is Callable:
        Additional to `other_context`, the optimizer is returned as first element of the tuple.

TODO: 
CUDA still raises warning of "producer process termintated before all shared tensors are released". yet the function seems operating properly, and all necessary tensor are cloned. Likely I leak some vairables that hold reference to the shared tensor before send ack to migration_save, but should not be a problem.
"""
def migration_load(
                   IPC_queues: Tuple[mp.Queue, mp.Queue],
                   model: Module,
                   optimizer: Optional[Union[Optimizer, Callable]] = None,
                   device: torch.device = None,
                   world: Optional[dist.ProcessGroup] = None,
                   log_func: Optional[Callable] = None,
                   checkpoint_resume: Optional[str] = None,
    ) -> Union[Tuple[Optimizer, Union[Any, None]], Union[Any, None]]:
  if log_func is None:
    log_func = lambda msg: None
  if device is None:
    device = torch.get_default_device()
  elif not isinstance(device, torch.device):
    device = torch.device(device)
  if device.type == "cuda":
    torch.cuda.set_device(device.index)
  log_func(f"migration_load: begin")
  
  
  if checkpoint_resume is not None:
    # use traditional checkpoint-resume mechanism
    log_func(f"migration_load: checkpoint_resume = {checkpoint_resume}")
    return checkpoint_load(
      model=model,
      optimizer=optimizer,
      checkpoint_dir=checkpoint_resume,
      world=world,
      log_func=log_func
    )
  
  if isinstance(optimizer, Callable):
    optimizer_provider = optimizer
    del optimizer
  elif isinstance(optimizer, Optimizer) or optimizer is None:
    optimizer_provider = None
  else:
    raise TypeError(f"optimizer should be Optimizer or Callable, but got {type(optimizer)}")
  
  prof = Profiler(strategy="last", device=device)
  
  # handshake with migration_save to determine if is_scale_up
  rank = dist.get_rank(world)
  if rank == 0:
    # avoid deadlock
    old_world_size = IPC_queues[0].get(block = True)
    new_world_size = dist.get_world_size(world)
    if old_world_size != 0:
      IPC_queues[1].put(new_world_size)
    is_scale_up = new_world_size >= old_world_size
  else:
    is_scale_up = None
  is_scale_up: bool = broadcast_objects(is_scale_up, group=world)
    
  log_func(f"migration_load: is_scale_up = {is_scale_up}")
  
  # get target_param_layout
  log_func(">> before param layout")
  prof.start("param_layout")
  target_param_layout = ParamLayout()
  target_param_layout.gather(model, world)
  prof.stop("param_layout")
  log_func(">> after param layout")
  
  if is_scale_up:
    # scale up, schedule and execute param transfer here
    
    # prepare sources
    log_func(">> before recv")
    prof.start("recv", require_memory=True)
    
    # a dummy IPC_state_dict is padded for freshly launched process, so the queue is not empty
    IPC_state_dict = interactive_IPC_state_dict_recv(IPC_queues=IPC_queues)
    source_param_layout = IPC_state_dict["source_param_layout"]
    source_state_dict = IPC_state_dict["source_state_dict"]
    other_context_dict = IPC_state_dict["other_context_dict"]
    IPC_state_dict.clear()
    del IPC_state_dict
    cuda_gc()

    prof.stop("recv", require_memory=True)
    log_func(">> after recv")
    
    log_func(">> before broadcast")
    prof.start("broadcast")
    source_param_layout, other_context_dict = broadcast_objects(
      source_param_layout, other_context_dict, group=world
    )
    prof.stop("broadcast")
    log_func(">> after broadcast")
    
    # the job is freshly launched, no checkpoint received
    if source_param_layout is None:
      log_func(">> no checkpoint")
    
      log_func(f"migration_load: no checkpoint received; skip loading")

      log_func(">> before instantiate")      
      if hasattr(model, "instantiate") and getattr(model, "on_meta", False):
        model.instantiate()
      
      log_func(">> after instantiate")

      if optimizer_provider:
        return optimizer_provider(), None
      else:
        return None
  
    log_func(">> before transfer")
    prof.start("transfer", require_memory=True)
    target_state_dict = schedule_and_execute_param_transfer(
      source_param_layout=source_param_layout,
      target_param_layout=target_param_layout,
      source_state_dict=source_state_dict,
      other_context_dict=other_context_dict,
      device=device,
      world=world,
      prof=prof,
      forcefully_free=False,
      log_func=log_func
    )
    del source_state_dict
    prof.stop("transfer", require_memory=True)
    log_func(">> after transfer")
    
    cuda_gc()    
    dist.barrier(world)
  else:
    # not scale up, send back target_param_layout (rank 0 only), accept target_state_dict from IPC_queues, then clone
    
    # send target_param_layout to migration_save
    if rank == 0:
      IPC_queues[1].put(target_param_layout)
    
    # receive target_state_dict and other_context_dict from migration_save
    log_func(">> before recv")
    prof.start("recv", require_memory=True)
    IPC_state_dict = interactive_IPC_state_dict_recv(IPC_queues=IPC_queues)
    target_state_dict = IPC_state_dict["target_state_dict"]
    other_context_dict = IPC_state_dict["other_context_dict"]
    IPC_state_dict.clear()
    del IPC_state_dict
    cuda_gc()
    prof.stop("recv", require_memory=True)
    log_func(">> after recv")

    dist.barrier(world)

  # split model state dict and optimizer state dict
  log_func(">> before split")
  prof.start("split", require_memory=True)
  model_state_dict, optimizer_state_dict, other_context = \
    split_model_and_optimizer_state_dict(model=model, merged_state_dict=target_state_dict, other_context_dict=other_context_dict)
  prof.stop("split", require_memory=True)
  log_func(">> after split")
  
  # load model state dict
  log_func(">> before load")
  prof.start("load", require_memory=True)

  model_state_dict = OrderedDict(sorted(model_state_dict.items(), key=lambda x: x[0])) # ensure consistent parram register order
  model.load_state_dict(model_state_dict, assign=True)

  # load optimizer state dict
  if optimizer_provider:
    optimizer = optimizer_provider()
  if optimizer:
    # this is a hacky workaround to resolve change of param number
    # multi param_groups optimizer not supported (multi param_groups is not likely in our use case anyway)
    optimizer_state_dict["param_groups"][0]["params"] = optimizer.state_dict()["param_groups"][0]["params"]
    optimizer.load_state_dict(optimizer_state_dict)

  prof.stop("load", require_memory=True)
  log_func(">> after load")
    
  log_func(f"migration_load: {prof.full_report_str(require_memory=True)}")
  log_func(f"migration_load: end")
  
  
  if optimizer_provider:
    return optimizer, other_context
  else:
    return other_context