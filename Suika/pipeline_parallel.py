import torch
from torch import Tensor
from torch.nn import Module
import torch.distributed as dist
from torch.distributed import P2POp

import logging
from logging import Logger

from typing import Dict, Optional, List, Tuple, Callable

from Suika.device_mesh import DeviceMesh2D, DeviceMesh3D
from Suika.profiler import Profiler
from Suika.utils import instantiate_non_persistent_buffers, maybe_batch_isend_irecv, reset_parameters


class PipelineExecModule(Module):
  """
  The PipelineExecModule class represents a pipeline execution unit in a distributed pipeline parallelism setup.
  It is responsible for managing the execution of a partitioned model across multiple stages of the pipeline.

  Attributes:
  - model (Module): The partitioned model held by this stage.
  - device (torch.device): The device on which the current process is running.
  - mesh (DeviceMesh3D): The 3D device mesh representing the pipeline stages.
  - lbsz_mesh (List[List[int]]): The local batch size mesh.
  - enable_grad_sync (Optional[Callable]): Function to enable gradient synchronization at data parallel (DP) level.
  - disable_grad_sync (Optional[Callable]): Function to disable gradient synchronization at DP level.
  - grad_sync_is_enabled (bool): Indicates whether gradient synchronization is currently enabled.
  - world (Optional[dist.ProcessGroup]): The process group for communication.
  - rank (int): The global rank of the current process.
  - recv_input_op (Callable): Function to receive input tensors.
  - send_output_op (Callable): Function to send output tensors.
  - recv_output_grad_op (Callable): Function to receive output gradients.
  - send_input_grad_op (Callable): Function to send input gradients.
  - pp_rank (int): The pipeline parallel (PP) index of the current stage.
  - dp_rank (int): The data parallel (DP) index of the current stage.
  - tp_rank (int): The tensor parallel (TP) index of the current stage.
  - lbsz (int): The local micro batch size.
  - dtype (torch.dtype): The data type of the tensors.
  - input_shape (List[int]): The shape of the input tensors.
  - output_shape (List[int]): The shape of the output tensors.
  - batch_dim (int): The dimension along which batches are divided.
  - slice_dim (int): The dimension along which tensors are sliced.
  - input_buffer (List[Tensor]): Buffer to store input tensors.
  - output_buffer (List[Tensor]): Buffer to store output tensors.
  - p2pop_buffer (List[P2POp]): Buffer to store P2P operations of one time index
  - logger (Optional[Logger]): Logger for logging messages.
  - logging_pattern (str): Pattern for logging messages.
  - prof (Profiler): Profiler for performance measurement.
  - enable_profiling (bool): Indicates whether profiling is enabled.
  """
  # PipelineExec is like static compiled pipeline execution unit
  
  # model
  model: Module # only partitioned model hold by this stage - interleaved 1F1B is not supported
  
  # device
  mesh: DeviceMesh3D
  device: torch.device # device of current process
  on_meta: bool
  
  # lbsz
  lbsz_mesh: List[List[int]] # local batch size mesh
  
  # enable/disable sync
  enable_grad_sync: Optional[Callable]
  disable_grad_sync: Optional[Callable]
  grad_sync_is_enabled: bool # whether grad_sync is currently enabled (used to indicate last micro batch, and force sync barrier for async p2p ops)

  # commu
  world: Optional[dist.ProcessGroup]
  rank: int # global rank of current process

  recv_input_op: Callable # tensor -> None, perform recv input
  send_output_op: Callable

  recv_output_grad_op: Callable
  send_input_grad_op: Callable
  
  # coord in entire pipleline
  pp_rank: int
  dp_rank: int
  tp_rank: int
  
  # input/output tensor spec
  lbsz: int # local micro batch size
  
  input_dtype: torch.dtype
  output_dtype: torch.dtype
  input_shape: List[int]
  output_shape: List[int]
  
  batch_dim: int = 0
  slice_dim: int = -1
  
  # buffers
  input_buffer: List[Tensor]
  output_buffer: List[Tensor]
  p2pop_buffer: List[P2POp]
  
  # logger
  logger: Optional[Logger]
  logging_pattern: str
  
  # profiler
  prof: Profiler
  enable_profiling: bool
    
  def __init__(self,
               model: Module, # should be partitioned already, by, say PPEngine
               device: torch.device,
               mesh: DeviceMesh3D,
               lbsz_mesh: List[List[int]],
               enable_grad_sync: Optional[Callable] = None, # function to enable gradient sync at DP
               disable_grad_sync: Optional[Callable] = None, # function to disable gradient sync at DP
               initial_grad_sync_is_enabled: bool = False, # whether to enable grad_sync at init
               batch_dim: int = 0,
               slice_dim: int = -1,
               init_on_meta: bool = False, 
               world: Optional[dist.ProcessGroup] = None, # suggested to leave it to None, unless you need multiple pipeline
               logger: Optional[Logger] = None,
               profiler: Optional[Profiler] = None,
               enable_profiling: bool = True,
              ):
    super(PipelineExecModule, self).__init__()
    
    rank = dist.get_rank(world)
    coord = mesh.get_coord(rank)
    assert coord is not None, f"rank {rank} not in mesh {mesh}"
    
    #logger & profiler
    self.logger = logger
    self.logging_pattern = f"PEM[{rank}]@{coord}-{{}}"
        
    self.prof = Profiler(pattern=f"rank-{rank}-at-{coord}-{{}}", strategy="average_skip_first", device = device) \
                if profiler is None else \
                profiler
    self.enable_profiling = enable_profiling

    self.log("__init__")
    self.prof.start("PEM-__init__", True, False, self.enable_profiling)
    
    # model
    self.model = model if init_on_meta else model.to(device)
    
    # device
    self.mesh = mesh
    self.device = torch.device(device)
    self.on_meta = init_on_meta
    
    # lbsz
    self.lbsz_mesh = lbsz_mesh
    
    # enable/disable sync
    self.enable_grad_sync = enable_grad_sync
    self.disable_grad_sync = disable_grad_sync
    self.grad_sync_is_enabled = initial_grad_sync_is_enabled
    self.set_grad_sync(on_or_off = initial_grad_sync_is_enabled)
    
    # commu
    self.world = world
    self.rank = dist.get_rank(world)
    self.recv_input_op = None
    self.send_output_op = None
    self.recv_output_grad_op = None
    self.send_input_grad_op = None
    
    # coord
    self.pp_rank = coord[0]
    self.dp_rank = coord[1]
    self.tp_rank = coord[2]
    
    # input tensor spec
    self.lbsz = lbsz_mesh[self.pp_rank][self.dp_rank]
    
    self.input_dtype = None
    self.output_dtype = None
    self.input_shape = None
    self.output_shape = None
    
    self.batch_dim = batch_dim
    self.slice_dim = slice_dim
    
    # buffers
    self.input_buffer = []
    self.output_buffer = []
    
    self.p2pop_buffer = []
    
    # setup communication ops
    self._set_commu_ops()
    
    self.prof.stop("PEM-__init__", True, False, self.enable_profiling)
    self.log("__init__ complete")
    
  """
  @PUBLIC MAIN COMPONENT
  """
  
  """
  forward
  """
  def forward(self, input: Optional[Tensor] = None, mb_index: Optional[int] = None, *args, **kwargs):
    assert not self.on_meta, "model must be instantiated at forward"

    # sync DP group
    # self.model._sync_barrier()
    
    self.log("forward")
    self.prof.start("PEM-forward", True, False, self.enable_profiling)
    
    ## infer input/output shape is not initialized
    ## deprecated
    # self._infer_shape_forward(input)
    
    # infer shape forward before recv input
    self._infer_shape_input(input)

    # prepare input
    self.prof.start("PEM-forward-prepare-input", True, False, self.enable_profiling)
    if self.is_first_stage():
      assert input is not None, "input should be provided for first stage"
      input = input.to(self.device)

      # perform send from last forward/backword
      self._perform_p2pops(first_send_then_recv=True)
    else:
      if input is not None:
        self.log("input should be None for non-first stage. ignored", lvl = logging.WARN)
      
      # recv input tensor
      input = torch.zeros(self.input_shape, device=self.device, dtype=self.input_dtype)
      # recv will implicitly call _perform_p2pops
      self.recv_input_op(input) 

      # set input tensor to leaf node      
      input.requires_grad = True
      
      self.input_buffer.append(input)
    self.prof.stop("PEM-forward-prepare-input", True, False, self.enable_profiling)
    
    
    # forward compute
    measure_memory = mb_index == 0
    if measure_memory and self.device.type == "cuda":
      torch.cuda.reset_peak_memory_stats(self.device)

    self.prof.start("PEM-forward-compute", True, measure_memory, self.enable_profiling)
    if mb_index is not None:
      self.prof.start(f"PEM-forward-mb-{mb_index}", True, False, self.enable_profiling)
      
    output = self.model(input, *args, **kwargs)
    
    self.prof.stop("PEM-forward-compute", True, measure_memory, self.enable_profiling)
    if mb_index is not None:
      self.prof.stop(f"PEM-forward-mb-{mb_index}", True, False, self.enable_profiling)
      
    
    # save output
    self.output_buffer.append(output)
    
    # infer shape output before send output
    self._infer_shape_output(output)
  
    # send output
    self.prof.start("PEM-forward-send-output", True, False, self.enable_profiling)
    if not self.is_last_stage():
      self.send_output_op(output)
    self.prof.stop("PEM-forward-send-output", True, False, self.enable_profiling)

    # # perform send for last micro batch if in eval mode as there's no following op
    # #   in train mode, following backward will perform p2pops
    # if self.grad_sync_is_enabled and not self.training: 
    #   self._perform_p2pops()
    
    # sync DP group
    # self.model._sync_barrier()
    
    self.prof.stop("PEM-forward", True, False, self.enable_profiling)
    self.log("forward complete")
    
    return output if self.is_last_stage() else None
    
  """
  backward
  """
  def backward(self, label: Optional[Tensor] = None, loss_fn: Optional[Module] = None, mb_index: Optional[int] = None) -> Optional[Tensor]:
    assert not self.on_meta, "model must be instantiated at backward"
    
    # sync DP group
    # self.model._sync_barrier()
    
    self.log("backward")
    self.prof.start("PEM-backward", True, False, self.enable_profiling)
  
    # prepare ouput grad
    self.prof.start("PEM-backward-prepare-output-grad", True, False, self.enable_profiling)
    if self.is_last_stage():
      assert label is not None and loss_fn is not None, f"label({type(label)}) and loss_fn({type(loss_fn)}) should be provided for last stage"
      # perform send from last forward/backword
      self._perform_p2pops(first_send_then_recv=False)
    else:
      if label is not None or loss_fn is not None:
        self.log("label and loss_fn should be None for non-last stage. ignored", lvl = logging.WARN)
      
      output_grad = torch.zeros(self.output_shape, device=self.device, dtype=self.output_dtype)
      # recv output grad tensor
      # recv will implicitly call _perform_p2pops
      self.recv_output_grad_op(output_grad)
    self.prof.stop("PEM-backward-prepare-output-grad", True, False, self.enable_profiling)
    
    # prepare output
    output = self.output_buffer.pop(0)

    # backward compute
    self.prof.start("PEM-backward-compute", True, False, self.enable_profiling)
    if mb_index is not None:
      self.prof.start(f"PEM-backward-mb-{mb_index}", True, False, self.enable_profiling)
    
    self.log("begin backward compute")
    
    loss = None
    if self.is_last_stage():
      loss = loss_fn(output, label)
      loss.backward()
    else:
      output.backward(output_grad)
    output = None
    output_grad = None
  
    self.log("end backward compute")

    self.prof.stop("PEM-backward-compute", True, False, self.enable_profiling)
    if mb_index is not None:
      self.prof.stop(f"PEM-backward-mb-{mb_index}", True, False, self.enable_profiling)
  
    # send input grad
    self.prof.start("PEM-backward-send-input-grad", True, False, self.enable_profiling)
    if not self.is_first_stage():
      self.log("send input grad")

      input = self.input_buffer.pop(0)
      input_grad = input.grad
      self.send_input_grad_op(input_grad)
    else:
      self.log("send no input grad")

    self.prof.stop("PEM-backward-send-input-grad", True, False, self.enable_profiling)
      
    
    # synchronize for last micro batch (after sending input_grad)
    if self.grad_sync_is_enabled:
      # perform send for last micro batch before perform_sync_grad, avoiding blocking critical path
      # self._perform_p2pops(first_send_then_recv=False)
      self.sync()
      
      # self.prof.start("PEM-backward-compute", True, False, self.enable_profiling, attach_to_last = True)
      if mb_index is not None:  
        self.prof.start(f"PEM-backward-mb-{mb_index}", True, False, self.enable_profiling, attach_to_last = True)
      self.prof.start("PEM-grad-sync", True, False, self.enable_profiling)
      
      self.model._perform_sync_grad() # sync DP grad all reduce
      
      # self.prof.stop("PEM-backward-compute", True, False, self.enable_profiling, attach_to_last = True)
      if mb_index is not None:  
        self.prof.stop(f"PEM-backward-mb-{mb_index}", True, False, self.enable_profiling, attach_to_last = True)
      self.prof.stop("PEM-grad-sync", True, False, self.enable_profiling)
    
    # sync DP group
    # self.model._sync_barrier()
    
    self.prof.stop("PEM-backward", True, False, self.enable_profiling)
    self.log("backward complete")
    
    return loss if self.is_last_stage() else None
  
  def sync(self):
    if not len(self.p2pop_buffer):
      return
    
    self.log("sync")
    self.prof.start("PEM-sync", True, False, self.enable_profiling)
    
    # perform send from last forward/backword
    # when sync is called, it always holds only send, so first_send_then_recv does not matter
    self._perform_p2pops(first_send_then_recv=True)
    
    self.prof.stop("PEM-sync", True, False, self.enable_profiling)
    self.log("sync complete")
    
  """
  @PUBLIC UTILS
  """
  
  def instantiate(self, *args, **kwargs):
    self.log("instantiate")
    if self.on_meta:
      if hasattr(self.model, "instantiate"):
        self.model.instantiate(*args, **kwargs)
      else:
        self.model = self.model.to_empty(device=self.device)
        reset_parameters(self)
        
      self._post_instantiate(*args, **kwargs)
    else:
      self.log("already instantiated")
  
  def _post_instantiate(self, *args, **kwargs):
    if not self.on_meta:
      return 

    if hasattr(self.model, "_post_instantiate"):
      self.model._post_instantiate(*args, **kwargs)
    instantiate_non_persistent_buffers(self, self.device)
    
    self.on_meta = False

  def is_first_stage(self) -> bool:
    return self.pp_rank == 0
    
  def is_last_stage(self) -> bool:
    return self.pp_rank == self.mesh.pp_deg() - 1  
  
  def log(self, msg: str, lvl = logging.DEBUG):
    if self.logger is not None:
      msg_wrapped = self.logging_pattern.format(msg)
      self.logger.log(level = lvl, msg = msg_wrapped)

  def set_grad_sync(self, on_or_off: Optional[bool]):
    if on_or_off is None:
      return
    self.grad_sync_is_enabled = on_or_off
    if on_or_off:
      if self.enable_grad_sync is not None:
        self.enable_grad_sync()
    else:
      if self.disable_grad_sync is not None:
        self.disable_grad_sync()
    return
    
  def get_coord(self)->Tuple[int, int, int]:
    return (self.pp_rank, self.dp_rank, self.tp_rank)
  
  
  """
  OVERRIDE state_dict & named_parameters()
  """
    
  # skip self.model wrapper, directly access underlying model
  def state_dict(self, destination: Dict = None, prefix: str = "", keep_vars: bool = False):
    kwargs = {
      "destination": destination,
      "prefix": prefix,
      "keep_vars": keep_vars
    }
    
    return self.model.state_dict(**kwargs)
  
  
  def load_state_dict(self, state_dict, strict = True, assign = True):
    kwargs = {
      "state_dict": state_dict,
      "strict": strict,
      "assign": assign
    }
    
    if not assign and self.on_meta:
      self.instantiate(skip_broadcast_initial_param=True)
    result = self.model.load_state_dict(**kwargs)
    if assign and self.on_meta:
      self._post_instantiate(skip_broadcast_initial_param=True)
    return result
  
  
  def named_parameters(self, prefix = "", recurse = True, remove_duplicate = True):
    kwargs = {
      "prefix": prefix,
      "recurse": recurse,
      "remove_duplicate": remove_duplicate
    }
    return self.model.named_parameters(**kwargs)

  """
  @PRIVATE METHODS
  """

  # sync all tracked pending p2p ops
  def _sync_reqs(self):
    self.prof.start("PEM-_sync_reqs", True, False, self.enable_profiling)
    for req in self.req_handles_buffer:
      req.wait()
    self.req_handles_buffer = []
    
    self.prof.stop("PEM-_sync_reqs", True, False, self.enable_profiling)
  
  # add p2p ops to track
  def _track_p2pops(self, p2pops: List[P2POp]):
    self.p2pop_buffer.extend(p2pops)
    
  # issue batched p2p ops and wait
  # perform post p2p ops operations
  def _perform_p2pops(self, first_send_then_recv: bool):
    # forward: first send then recv (True)
    # backward: first recv then send (False)
    # ensure order consistency

    self.log(f"perform p2pops {len(self.p2pop_buffer)} / send {'>' if first_send_then_recv else '<'} recv")
    
    # note that this profiling includs waiting time, thus does not reflect accurate communication time.
    self.prof.start("PEM-_perform_p2pops", True, False, self.enable_profiling)
    if len(self.p2pop_buffer) > 0:
      # issue p2p ops
      # fusion of send/recv ops to avoid looped deadlock in 1F1B's last stage.

      send_ops = [p2pop for p2pop in self.p2pop_buffer if p2pop.op == dist.isend]
      recv_ops = [p2pop for p2pop in self.p2pop_buffer if p2pop.op == dist.irecv]
      reordered_p2pops = send_ops + recv_ops if first_send_then_recv else recv_ops + send_ops

      reqs = maybe_batch_isend_irecv(reordered_p2pops)
      for req in reqs:
        req.wait()      
      self.p2pop_buffer = []
    self.prof.stop("PEM-_perform_p2pops", True, False, self.enable_profiling)
    
    self.log("perform p2pops complete")
    return
  
  # initialize 4 communication ops given current mesh/rank info
  def _set_commu_ops(self):
    def do_nothing(target: Tensor):
      self.log("this communication method should not be called, check your code!!!", lvl = logging.FATAL)
      return
    
    prev_rank, self_rank, next_rank = self.pp_rank - 1, self.pp_rank, self.pp_rank + 1
    mesh_self = self.mesh[self_rank]
    lbszs_self = self.lbsz_mesh[self_rank]
    
    # output direction
    if self.is_last_stage():
      self.recv_output_grad_op = do_nothing
      self.send_output_op = do_nothing
    else:
      mesh_next = self.mesh[next_rank]
      lbszs_next = self.lbsz_mesh[next_rank]
      
      self.recv_output_grad_op = self._get_recv_op(
        rank_dst = self.rank, 
        mesh_src = mesh_next, mesh_dst = mesh_self, 
        local_bszs_src = lbszs_next, local_bszs_dst = lbszs_self,
        batch_dim = self.batch_dim, slice_dim = self.slice_dim,
        first_send_then_recv=False, # backward
        world = self.world,
        log_func = self.log
      )
      self.send_output_op = self._get_send_op(
        rank_src = self.rank, 
        mesh_src = mesh_self, mesh_dst = mesh_next, 
        local_bszs_src = lbszs_self, local_bszs_dst = lbszs_next,
        batch_dim = self.batch_dim, slice_dim = self.slice_dim,
        world = self.world,
        log_func = self.log
      )

    # input direction
    if self.is_first_stage():
      self.recv_input_op = do_nothing
      self.send_input_grad_op = do_nothing
    else:
      mesh_prev = self.mesh[prev_rank]
      lbszs_prev = self.lbsz_mesh[prev_rank]
      self.recv_input_op = self._get_recv_op(
        rank_dst = self.rank, 
        mesh_src = mesh_prev, mesh_dst = mesh_self, 
        local_bszs_src = lbszs_prev, local_bszs_dst = lbszs_self,
        batch_dim = self.batch_dim, slice_dim = self.slice_dim,
        first_send_then_recv=True, # forward
        world = self.world,
        log_func = self.log
      )
      self.send_input_grad_op = self._get_send_op(
        rank_src = self.rank,
        mesh_src = mesh_self, mesh_dst = mesh_prev,
        local_bszs_src = lbszs_self, local_bszs_dst= lbszs_prev,
        batch_dim = self.batch_dim, slice_dim = self.slice_dim,
        world = self.world,
        log_func = self.log
      )
    return
  
  def _infer_shape_input(self, input: Optional[Tensor] = None):
    if self.input_shape is not None:
      return

    self.log("_infer_shape_input")
    
    # prepare input tensor
    if self.is_first_stage(): # first stage, input provided
      input = input.to(self.device)
      self.input_dtype = input.dtype
      self.input_shape = list(input.size())   
    else: # not first stage, receive from last stage
      self._recv_input_shape_from_prev_stage() # input shape/dtype is set inside the helper function

    return
    
  def _infer_shape_output(self, output: Tensor):
    if self.output_shape is not None:
      return

    self.log("_infer_shape_output")
    self.output_dtype = output.dtype
    self.output_shape = list(output.size())
    
    # send output tensor shape to next stage
    if not self.is_last_stage() and self.rank == self.mesh[self.pp_rank].get_rank_by_group(0, 0):
      # we let the first machine in the entire DP group to broadcast the output shape to next stage as input shape
      self._broadcast_output_shape_to_next_stage()
      
    return
    
  # helper function for _infer_shape_forward
  def _broadcast_output_shape_to_next_stage(self):
    self.log("_broadcast_output_shape_to_next_stage")
    
    shape = list(self.output_shape)
    shape[self.batch_dim] = -1
    dtype = self.output_dtype
    
    for rank_dst in self.mesh[self.pp_rank + 1].flatten():
      object_list = [shape, dtype]
      dist.send_object_list(dst=rank_dst, object_list=object_list, group=self.mesh.get_cpu_group())
      self.log(f"send shape to {rank_dst}:\n {object_list} \n")
    
    self.log(f"broadcast shape: complete")
    return
  
  # helper function for _infer_shape_forward
  def _recv_input_shape_from_prev_stage(self):
    self.log("_recv_input_shape_from_prev_stage")
    
    rank_src = self.mesh[self.pp_rank - 1].get_rank_by_group(0, 0)

    object_list = [None, None]
    dist.recv_object_list(src=rank_src, object_list=object_list, group=self.mesh.get_cpu_group())
    
    shape, dtype = object_list
    shape[self.batch_dim] = self.lbsz # set batch_dim to local batch size, which was -1 at transmission
    
    self.input_shape = shape
    self.input_dtype = dtype
    
    self.log(f"recv shape from {rank_src}:\n {object_list} \n")
    
    self.log(f"recv shape: complete")
    return
  
  """
  @STATIC METHODS
  Generate static method that only takes in one tensor as parameter to perform cross stage communications
  """

  # given two DP groups in adjacent stages, generate a (tensor -> List[P2POp]) callable to represent needed send/recv p2p opeartion
  # the core function of communication compilation
  @staticmethod
  def _get_recv_or_send_P2P_ops(
                         rank: int, 
                         mesh_src: DeviceMesh2D, mesh_dst: DeviceMesh2D, 
                         local_bszs_src: List[int], local_bszs_dst: List[int],
                         op: Callable, # choice : [dist.isend, dist.irecv]
                         batch_dim: int = 0,
                         slice_dim: int = -1, 
                         world: Optional[dist.ProcessGroup] = None,
                         log_func: Optional[Callable] = None, # should be Logger.info, debug, etc., or self.log func(str) -> None
                         ) -> Callable: 
    # we align P2P communication as the same in DP group communication
    # and perform send/recv if.f. two group overlaps in batch dimension (sliced at batch_dim)
    # for sender, only P2P is needed
    # for recver, first perform P2P to collect its own partition (sliced at slice_dim), 
    #             then all_gather in TP group
    
    if op == dist.isend:
      mesh_dst, mesh_src = mesh_src, mesh_dst
      local_bszs_src, local_bszs_dst = local_bszs_dst.copy(), local_bszs_src.copy()
    elif op == dist.irecv:
      pass
    else:
      assert False, f"op should be dist.isend or dist.irecv, but got {op}"
    
    # following code assume recv    
    rank_dst = rank
    
    coord = mesh_dst.get_coord(rank_dst)
    assert coord is not None, f"rank {rank} not in mesh {mesh_dst}"
    (dst_dp, dst_tp) = coord
    
    dst_tp_deg = mesh_dst[dst_dp].tp_deg() # the actual tp deg of the dst tp group, no virtualization alignment
    
    batch_range_src = [(sum(local_bszs_src[:i]), sum(local_bszs_src[:i+1])) for i in range(mesh_src.dp_deg())]
    batch_range_dst = [(sum(local_bszs_dst[:i]), sum(local_bszs_dst[:i+1])) for i in range(mesh_dst.dp_deg())]
    
    p2p_op_gens = []
      
    for src_dp in range(mesh_src.dp_deg()):
      # compute batch overlap
      (l_src, r_src) = batch_range_src[src_dp]
      (l_dst, r_dst) = batch_range_dst[dst_dp]
      (l_olp, r_olp) = (max(l_src, l_dst), min(r_src, r_dst))
      
      # log_func(f"batch overlap btw {src_dp} {dst_dp} = [{l_olp}:{r_olp}]")

      if l_olp >= r_olp:
        continue

      src_tp_deg = mesh_src[src_dp].tp_deg()        
      send_begin = dst_tp * src_tp_deg // dst_tp_deg
      
      if src_tp_deg >= dst_tp_deg: # src many to dst 1
        send_end = (dst_tp + 1) * src_tp_deg // dst_tp_deg 
      else: # src 1 to dst many
        send_end = send_begin + 1
        
      for src_tp in range(send_begin, send_end):
        rank_src = mesh_src[src_dp][src_tp]
        
        if src_tp_deg >= dst_tp_deg:
          tp_deg = src_tp_deg
          tp_rank = src_tp
        else:
          tp_deg = dst_tp_deg
          tp_rank = dst_tp
                
        def p2p_op_gen(target: torch.Tensor, context = (l_olp, r_olp, l_dst, rank_src, rank_dst, tp_deg, tp_rank, world)): # context argument is only for static capture purpose
          
          l_olp, r_olp, l_dst, rank_src, rank_dst, tp_deg, tp_rank, world = context
          
          # batch_dim chunk
          target_batch_chunked = target.narrow(dim = batch_dim, start = l_olp - l_dst, # offset into local coord
                                               length = r_olp - l_olp)
        
          # slice_dim chunk
          chunk_size = target_batch_chunked.size(slice_dim) // tp_deg  
          # target_slice_chunked = target_batch_chunked.chunk(tp_deg, dim = slice_dim)[tp_rank]
          target_slice_chunked = target_batch_chunked.narrow(dim = slice_dim, start = tp_rank * chunk_size, length = chunk_size)

          ## logging p2p op
          # if op == dist.irecv:
          #   actual_rank_src, actual_rank_dst = rank_src, rank_dst
          # else:
          #   actual_rank_src, actual_rank_dst = rank_dst, rank_src  
          # log_func(f"{actual_rank_src} to {actual_rank_dst} {'send' if op == dist.isend else 'recv'} [{l_olp}:{r_olp}][{tp_rank * chunk_size}:{(tp_rank + 1) * chunk_size}]")
                        
          # due to the fact sliced target may be incontiguous, we have to reformat, and return a callable to restore this tensor to needed place
          transmit_tensor = target_slice_chunked.contiguous()
          def restore_transmit_tensor():
            target_slice_chunked.copy_(transmit_tensor)
          
          return (P2POp(op=op, peer = rank_src, tensor = transmit_tensor, group=world), restore_transmit_tensor)
        p2p_op_gens.append(p2p_op_gen)
        
    def P2P_ops_generator(target: torch.Tensor) -> Tuple[List[dist.P2POp], Callable]:
      p2p_ops = []
      restore_funcs = []
      for p2p_op_gen in p2p_op_gens:
        p2p_op, transmit_tensor = p2p_op_gen(target)
        p2p_ops.append(p2p_op)
        restore_funcs.append(transmit_tensor)
        
      def restore_transmit_tensor():
        for restore_func in restore_funcs:
          restore_func()
      return (p2p_ops, restore_transmit_tensor) 
    
    return P2P_ops_generator
    

  """
  @brief: given two DP groups in adjacent stages, generate a (tensor -> None) callable, that performs the recv operation (modification is in place)
  """
  def _get_recv_op(
    self,
    rank_dst: int, 
    mesh_src: DeviceMesh2D, 
    mesh_dst: DeviceMesh2D,
    local_bszs_src: List[int], 
    local_bszs_dst: List[int], 
    first_send_then_recv: bool,
    batch_dim: int = 0,
    slice_dim: int = -1,
    world: Optional[dist.ProcessGroup] = None,
    log_func: Optional[Callable] = None,
    ) -> Callable:
    if log_func is None:
      log_func = lambda msg: None
    
    (dst_dp, dst_tp) = mesh_dst.get_coord(rank_dst)
    
    dst_tp_deg = mesh_dst[dst_dp].tp_deg() # the actual tp deg of the dst tp group, no virtualization alignment
    
    recv_P2P_ops_generator = PipelineExecModule._get_recv_or_send_P2P_ops(rank_dst, mesh_src, mesh_dst, local_bszs_src, local_bszs_dst, 
                                                                            op = dist.irecv, 
                                                                            batch_dim = batch_dim, slice_dim = slice_dim, 
                                                                            world=world,
                                                                            log_func = log_func)
    
    def recv_op(target: torch.Tensor):
      p2p_ops, restore_func = recv_P2P_ops_generator(target)

      log_func(f"recv issue {len(p2p_ops)} transmission(s)")
      
      if len(p2p_ops):
        self._track_p2pops(p2p_ops)
        self._perform_p2pops(first_send_then_recv=first_send_then_recv)
        restore_func()
            
      log_func(f"recv p2p complete")
        
      if len(mesh_dst[dst_dp]) == 1:
        log_func(f"tp degree = 1, skip recv all gather")
      else:
        target_slice_chunked = target.chunk(dst_tp_deg, dim = slice_dim)[dst_tp]

        target_list = [torch.empty_like(target_slice_chunked) for _ in range(dst_tp_deg)]
        
        log_func(f"recv all_gather, size = {tuple(target_slice_chunked.size())}")
        # log_func(f"\n{target_slice_chunked}")
        
        dist.all_gather(target_list, target_slice_chunked.contiguous(), group = mesh_dst[dst_dp].get_tp_group())
        
        log_func(f"recv all_gather complete")
        # log_func(f"\n{target_list}")

        target.copy_(torch.cat(target_list, dim = slice_dim))
      
    return recv_op

  """
  @brief: given two DP groups in adjacent stages, generate a (tensor -> reqs) callable, that performs the send operation (sync handles will be returned by send_op)
  """
  def _get_send_op(
      self,
      rank_src: int,
      mesh_src: DeviceMesh2D,
      mesh_dst: DeviceMesh2D,
      local_bszs_src: List[int],
      local_bszs_dst: List[int],
      batch_dim: int = 0,
      slice_dim: int = -1,
      world: Optional[dist.ProcessGroup] = None,
      log_func: Optional[Callable] = None
  ) -> Callable:
    if log_func is None:
      log_func = lambda msg: None
    
    send_P2P_ops_generator = PipelineExecModule._get_recv_or_send_P2P_ops(
      rank_src, mesh_src, mesh_dst, local_bszs_src, local_bszs_dst,
      op = dist.isend,
      batch_dim=batch_dim, slice_dim=slice_dim, 
      world = world,
      log_func = log_func
    )
    def send_op(tensor: torch.Tensor):
      p2p_ops, _ = send_P2P_ops_generator(tensor) # we don't need restore func for send
      
      log_func(f"send issue {len(p2p_ops)} transmission(s)")
      
      self._track_p2pops(p2p_ops)
      return 
    
    return send_op



"""
Pipeline Execution Orchestration
"""

class PipelineExecOp:
  """
  The PipelineExecOp class represents an operation to be executed within a pipeline stage in a distributed pipeline parallelism setup.
  This class encapsulates the details of the operation, such as the stage index, the type of operation (forward or backward), and optional
  settings for gradient synchronization and micro-batch index.

  Attributes:
  - stage_index (int): The index of the pipeline stage where the operation should be executed.
  - op (str): The type of operation to be performed. It can be one of the following:
    - "fwd", "fw", or "forward": Indicates a forward pass operation.
    - "bwd", "bw", or "backward": Indicates a backward pass operation.
  - mb_index (int): The micro-batch index, used to identify the specific micro-batch being processed when loading data by PPEngine.
  - grad_sync (Optional[bool]): A flag indicating whether to enable or disable gradient synchronization at the beginning of the operation.
  - time_index (Optional[int]): The time index for the operation, used to identify the specific time step in the pipeline execution. (for visuialize a schedule for debug purpose)

  Methods:
  - __init__(self, stage_index: int, op: str, grad_sync: Optional[bool] = None, mb_index: Optional[int] = None):
    Initializes a new instance of the PipelineExecOp class with the specified stage index, operation type, gradient synchronization flag,
    and micro-batch index.

  - __call__(self, exec_module: PipelineExecModule, input_or_label: Optional[Tensor] = None, *args, loss_fn: Optional[Tensor] = None, **kwargs):
    Executes the operation on the given PipelineExecModule instance. Depending on the operation type (op), it calls the forward or backward
    method of the PipelineExecModule. If the current stage does not match the stage_index, the operation is skipped. Sets the gradient
    synchronization state before executing the operation.

  - is_forward(self) -> bool:
    Returns True if the operation is a forward pass, otherwise False.

  - is_backward(self) -> bool:
    Returns True if the operation is a backward pass, otherwise False.

  - skip(self, stage_index: int) -> bool:
    Returns True if the operation should be skipped for the given stage index, otherwise False.
  """
  
  stage_index: int # on which stage should the op be executed
  op: str # choice : ["fwd", "bwd"]

  grad_sync: Optional[bool] # whether to enable/disable grad_sync at the beginning of operation
  
  mb_index: Optional[int] # mirco batch index, useful for external code to fetch corresponding input

  def __init__(self, stage_index: int, op: str, mb_index:int, grad_sync: Optional[bool] = None,  time_index: Optional[int] = None):
    
    self.stage_index = stage_index
    self.op = op
    self.mb_index = mb_index
    self.grad_sync = grad_sync
    self.time_index = time_index
  
  # key function to PipelineExecOp
  def __call__(self, 
               exec_module: PipelineExecModule, 
               input_or_label: Optional[Tensor] = None, *args, 
               loss_fn: Optional[Tensor] = None, **kwargs):
    assert isinstance(exec_module, PipelineExecModule), f"exec_module should be PipelineExecModule, but got {type(exec_module)}"
    
    if self.skip(exec_module.get_coord()[0]):
      return # skip if not the target stage
    
    exec_module.set_grad_sync(self.grad_sync)
      
    if self.is_forward():
      return exec_module.forward(input = input_or_label, mb_index=self.mb_index, *args, **kwargs)
    elif self.is_backward():
      return exec_module.backward(loss_fn = loss_fn, label = input_or_label, mb_index=self.mb_index)
    elif self.is_sync():
      return exec_module.sync()
    else:
      assert False, f"op should be 'forward' or 'backward', but got {self.op}"
    
  def is_forward(self) -> bool:
    return self.op in ["f", "fw", "fwd", "forward"]
  def is_backward(self) -> bool:
    return self.op in ["b", "bw", "bwd", "backward"]
  def is_sync(self) -> bool:
    return self.op in ["s", "sync"]
  
  def skip(self, stage_index) -> bool:
    return self.stage_index != stage_index
  
  def __str__(self):
    result = f"({self.stage_index}, {self.op}"
    if self.grad_sync is not None:
      result += f", grad_sync={self.grad_sync}"
    if self.mb_index is not None:
      result += f", mb_index={self.mb_index}"
    result += ")"
    return result
    
  def __repr__(self):
    return str(self)
    
    
    
def insert_sync_ops(exec_ops: List[PipelineExecOp], num_stage: int, num_micro_batch: int):
  """
  """
  total_time = 2 * num_stage + 2 * (num_micro_batch - 1)
  fwd = [[None] * total_time for _ in range(num_stage)]
  bwd = [[None] * total_time for _ in range(num_stage)]
  
  for op in exec_ops:
    if op.is_forward():
      fwd[op.stage_index][op.time_index] = op.mb_index
    elif op.is_backward():
      bwd[op.stage_index][op.time_index] = op.mb_index
      
  # insert sync (flush) op, iff. there is immediate waiter at next time index
  for t in range(1, total_time):
    for s in range(num_stage):
      if fwd[s][t] is None and bwd[s][t] is None:        
        next_sync = total_time
        for t_next in range(t, total_time):
          if fwd[s][t_next] is not None or bwd[s][t_next] is not None:
            next_sync = t_next
            break
          
        next_wait = total_time
        if fwd[s][t - 1] is not None:
          f = fwd[s][t - 1]
          if s < num_stage - 1:
            for t_next in range(t, total_time):
              if fwd[s + 1][t_next] == f:
                next_wait = t_next
                break
        elif bwd[s][t - 1] is not None:
          b = bwd[s][t - 1]
          if s > 0:
            for t_next in range(t, total_time):
              if bwd[s - 1][t_next] == b:
                next_wait = t_next
                break
        
        if next_sync > next_wait:
          exec_ops.append(
            PipelineExecOp(s, "sync", mb_index = None, grad_sync = None, time_index = t)
          )
  
  exec_ops.sort(key = lambda x: (x.time_index, x.stage_index))
  return exec_ops

def schedule_pipeline_GPipe(num_stage: int, num_micro_batch: int):
  exec_ops = []
  # forward
  for mb in range(num_micro_batch):
    for stage in range(num_stage):
      exec_ops.append(
        PipelineExecOp(stage, "forward", mb_index = mb, grad_sync=False if mb == 0 else None, time_index=mb + stage)
      )
      
  # backward
  for mb in range(num_micro_batch):
    for stage in range(num_stage):
        exec_ops.append(
          PipelineExecOp(stage, "backward", mb_index = mb, grad_sync = True if mb == num_micro_batch - 1 else None, 
                         time_index=mb + (num_stage - stage - 1) + num_micro_batch + num_stage - 1)
        )
        
  return insert_sync_ops(exec_ops, num_stage, num_micro_batch)
  
def schedule_pipeline_1F1B_legacy(num_stage: int, num_micro_batch: int):
  exec_ops = []
    
  fwd_ready: List[List[int]] = [[] for i in range(num_stage)]
  next_fwd_ready: List[List[int]] = [[] for i in range(num_stage)]
  bwd_ready: List[List[int]] = [[] for i in range(num_stage)]
  next_bwd_ready: List[List[int]] = [[] for i in range(num_stage)]
  
  warmup = min(num_stage, num_micro_batch)
  cur_input_mb = 0
    
  for time_index in range(2 * num_stage + 2 * (num_micro_batch - 1)):
    for i in range(num_stage):
      fwd_ready[i].extend(next_fwd_ready[i])
      next_fwd_ready[i] = []
      bwd_ready[i].extend(next_bwd_ready[i])
      next_bwd_ready[i] = []
    
    if time_index < warmup or \
      (time_index >= 2 * num_stage and cur_input_mb < num_micro_batch): # warmup phase and steady phase
      fwd_ready[0].append(cur_input_mb)
      cur_input_mb += 1

    for stage in range(num_stage):
      if len(bwd_ready[stage]) > 0:
        # backward
        mb = bwd_ready[stage].pop(0)
        exec_ops.append(
          PipelineExecOp(stage, "backward", mb_index = mb, grad_sync = True if mb == num_micro_batch - 1 else None, time_index=time_index)
        )
        if stage > 0:
          next_bwd_ready[stage - 1].append(mb)
        
      elif len(fwd_ready[stage]) > 0:
        # forward
        mb = fwd_ready[stage].pop(0)
        exec_ops.append(
          PipelineExecOp(stage, "forward", mb_index = mb, grad_sync=False if mb == 0 else None, time_index=time_index)
        )
        
        # prepare next stage
        if stage + 1 < num_stage:
          next_fwd_ready[stage + 1].append(mb)
        else: # last stage
          next_bwd_ready[stage].append(mb)
      else:
        # sync, but not every sync need flush
        pass
  
  return insert_sync_ops(exec_ops, num_stage, num_micro_batch)

  
## This version is proper
# |F 0|   |F 1|   |F 2|B 0|   |B 1|   |B 2|   |   |   
# |   |F 0|   |F 1|B 0|F 2|B 1|   |B 2|   |   |   |   
# |   |   |F 0|B 0|F 1|B 1|F 2|B 2|   |   |   |   |  '

def schedule_pipeline_1F1B(num_stage: int, num_micro_batch: int):
  def forward_time_index(stage: int, mb_index: int):
    return stage + mb_index * 2
  def backward_time_index(stage: int, mb_index: int):
    return (num_stage - stage - 1) + mb_index * 2 + num_stage
  
  ops = []
  for stage in range(num_stage):
    for mb in range(num_micro_batch):
      ops.append(
        PipelineExecOp(stage, "forward", mb_index=mb, grad_sync=False if mb == 0 else None, time_index=forward_time_index(stage, mb))
      )
    
  for stage in range(num_stage):
    for mb in range(num_micro_batch):
      ops.append(
        PipelineExecOp(stage, "backward", mb_index=mb, grad_sync=True if mb == num_micro_batch - 1 else None, time_index=backward_time_index(stage, mb))
      )
      
  return insert_sync_ops(ops, num_stage, num_micro_batch)

"""
Generate a list of PipelineExecOp objects based on the specified scheduling strategy.

Args:
- strategy (str): The scheduling strategy to use. Choices are 'GPipe', '1F1B', or 'interleaved1F1B'.
- num_stage (int): The number of pipeline stages.
- num_micro_batch (int): The number of micro-batches.

Returns:
- List[PipelineExecOp]: A list of PipelineExecOp objects representing the execution schedule.
"""
def schedule_pipeline(strategy: str, # choice from GPipe / 1F1B / interleaved1F1B
                      num_stage: int,
                      num_micro_batch: int,
                      ) -> List[PipelineExecOp]:
  if strategy in ["GPipe", "gpipe"]:
    return schedule_pipeline_GPipe(num_stage, num_micro_batch)
  elif strategy in ["1F1B", "1f1b"]:
    return schedule_pipeline_1F1B(num_stage, num_micro_batch)
  elif strategy == "interleaved1F1B":
    raise NotImplementedError("interleaved1F1B is not supported yet")
  else:
    assert False, (f"strategy should be from ['GPipe', '1F1B', 'interleaved1F1B'], but got {strategy}")
    
    
    
    
    
    
    
    
    
    
    
    
    
"""
Pipeline Model Partition

Proven inpractical
Deprecated.
"""

class StageBoundaryMarker(Module):
  
  def __init__(self):
    super(StageBoundaryMarker, self).__init__()
    
  def forward(self, input: Tensor):
    return input
  
  
def pipeline_partition_by_stage_boundary_mark(module) -> List[Module]:
  pass