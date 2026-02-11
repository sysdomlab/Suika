from collections import OrderedDict
from contextlib import contextmanager
import torch

import torch.distributed as dist
from torch.nn import Module
from torch import Tensor

from typing import Any, Callable, Dict, Optional, List, Tuple, Union, Iterator
import logging
from logging import Logger

from Suika.pipeline_parallel import PipelineExecModule, PipelineExecOp, schedule_pipeline

from Suika.device_mesh import DeviceMesh1D, DeviceMesh2D, DeviceMesh3D
from Suika.tensor_parallel import parallelize_tp, set_tp_attr
from Suika.profiler import Profiler
from Suika.utils import maybe_batch_isend_irecv, instantiate_non_persistent_buffers, reset_parameters


class TPEngine(Module):
  
  #model
  model: Module
  
  #device
  device: torch.device
  mesh: DeviceMesh1D
  
  on_meta: bool
  
  #commu
  world: Optional[dist.ProcessGroup] = None
  rank: int
  tp_deg: int
  tp_rank: int
  group: dist.ProcessGroup
  
  #logger
  logger: Optional[Logger]
  logging_pattern: str
  
  #profiler
  prof: Profiler
  enable_profiling: bool
  
  def __init__(self, 
               model: Module, # model to be parallized; 
               device: torch.device, # physical device of current process
               mesh: DeviceMesh1D, # device mesh for TP
               slice_dim: str = "column", # TP slice dimention, either "row" or "column" or "none" (for model that already parallized)
               init_on_meta: bool = False, # whether to initialize model on meta device, default to False.
               world: Optional[dist.ProcessGroup] = None,
               logger: Optional[Logger] = None,
               profiler: Optional[Profiler] = None,
               enable_profiling: bool = True,
              ):
    super(TPEngine, self).__init__()
    
    rank = dist.get_rank(world)
    coord = (mesh.get_coord(rank),)
    tp_rank = coord[0]
    
    #logger & profiler
    # moved to begining of __init__
    self.logger = logger
    self.logging_pattern = f"TPE[{rank}]@{coord}-{{}}"
    
    self.prof = (Profiler(pattern=f"rank-{rank}-at-{coord}-{{}}", strategy="average_skip_first", device=device)
                 if profiler is None else profiler)
    self.enable_profiling = enable_profiling
    
    self.log("__init__")
    self.prof.start("TPE-__init__", True, False, self.enable_profiling)

    #model    
    model = parallelize_tp(
                          model=model, 
                          device=device, 
                          mesh=mesh, 
                          slice_dim = slice_dim, 
                          init_on_meta=init_on_meta,
                          profiler=self.prof, 
                          enable_profiling=enable_profiling,
                          )
    self.model = model if init_on_meta else model.to(device)
  
    #device
    self.device = torch.device(device)
    self.mesh = mesh
    
    self.on_meta = init_on_meta
    
    #commu
    self.world = world
    self.rank = rank
    self.tp_deg = len(self.mesh)
    self.tp_rank = tp_rank
    self.group = self.mesh.get_tp_group()
    
    self.prof.stop("TPE-__init__", True, False, self.enable_profiling)
    self.log("__init__ complete")
    
  def forward(self, input, *args, **kwargs):
    assert not self.on_meta, "model must be instantiated at forward"
    
    self.prof.start("TPE-forward", True, False, self.enable_profiling)
      
    input = input.to(self.device)
    output = self.model(input, *args, **kwargs)
    
    self.prof.stop("TPE-forward", True, False, self.enable_profiling)
      
    return output
  
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
    
    self.log("_post_instantiate")
    
    if hasattr(self.model, "_post_instantiate"):
      self.model._post_instantiate(*args, **kwargs)
    instantiate_non_persistent_buffers(self, self.device)

    set_tp_attr(self.model)
    self.on_meta = False

  def log(self, msg: str, lvl = logging.DEBUG):
    if self.logger is not None:
      msg_wrapped = self.logging_pattern.format(msg)
      self.logger.log(level = lvl, msg = msg_wrapped)
      
  
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
      self.instantiate()
    result = self.model.load_state_dict(**kwargs)
    if assign and self.on_meta:
      self._post_instantiate()
    return result
  
  
  def named_parameters(self, prefix = "", recurse = True, remove_duplicate = True):
    kwargs = {
      "prefix": prefix,
      "recurse": recurse,
      "remove_duplicate": remove_duplicate
    }
    return self.model.named_parameters(**kwargs)


# DPEngine is transparent wrapper, that is, no training code or model code modification required
class DPEngine(Module):
  
  #model
  model: Module
  
  #device
  device: torch.device
  mesh: DeviceMesh2D
  on_meta: bool
  
  #commu
  enable_sync: bool # whether to sync gradients at backward
  world: dist.ProcessGroup
  rank: int # global rank of current process, trivially dist.get_rank(), but cache for convenience
  
  dp_deg: int
  dp_rank: int # rank in dp group, namely, in which tp computation group
  dp_groups: List[dist.ProcessGroup] # dp groups
  
  #sync_grad
  ordered_trainable_param: Dict[torch.nn.Parameter, int] # rank of parameter (requires grad) to serialize all_reduce order
  param_grad_ready: Dict[torch.nn.Parameter, bool]
  auto_perform_sync_grad: bool

  grad_scale_ratio: float # scale ratio of gradient
  sync_grad_hook_handles: List[torch.utils.hooks.RemovableHandle] # hooks 
  
  async_handles: List[dist.Work]
  
  #logger
  logger: Optional[Logger] # logger
  logging_pattern: str
  
  #profiler
  prof: Profiler
  
  def __init__(self, 
              model: Module, # model to be parallized, suggested to be on meta device
              device: torch.device, # physical device of current process
              mesh: DeviceMesh2D, # device mesh for DP + TP combined, first dim: dp, second dim: tp
              local_batch_sizes: Optional[List[int]] = None, # local batch sizes, if none, uniform grad_scale_ratio will be used
              auto_perform_sync_grad: bool = True, # whether _perform_sync_grad will be called automatically, set to False to allow decouple backward with sync_grad at pipeline parallelism
              tp_slice_dim: str = "column", # TP slice dimention, either "row" or "column" or "none" (for model that already parallized)
              init_on_meta: bool = False, 
              world: Optional[dist.ProcessGroup] = None,
              logger: Optional[Logger] = None,
              profiler: Optional[Profiler] = None,
              enable_profiling: bool = True,
              ):
    super(DPEngine, self).__init__()

    rank = dist.get_rank(world)
    coord = mesh.get_coord(rank)
    dp_rank = coord[0]
    
    #logger & profiler
    # moved to begining of __init__
    self.logger = logger
    self.logging_pattern = f"DPE[{rank}]@{coord}-{{}}"
    
    self.prof = (Profiler(pattern=f"rank-{rank}-at-{coord}-{{}}", strategy="average_skip_first", device=device)
                 if profiler is None else profiler)
    self.enable_profiling = enable_profiling
    
    self.log("__init__")
    self.prof.start("DPE-__init__", True, False, self.enable_profiling)    

    #model
    assert isinstance(model, Module), "DP Engine only support torch.nn.Module"
    model = TPEngine(
                    model=model, 
                    device=device, 
                    mesh=mesh[dp_rank],
                    slice_dim = tp_slice_dim,
                    init_on_meta=init_on_meta,
                    world=world,
                    logger=logger,
                    profiler=self.prof,
                    enable_profiling=enable_profiling,
                    )
    self.model = model if init_on_meta else model.to(device)
        
    #device
    assert isinstance(mesh, DeviceMesh2D), "DP Engine only support DeviceMesh2D"
    self.device = torch.device(device)
    self.mesh = mesh
    self.on_meta = init_on_meta
    
    #commu
    self.enable_sync = True
    
    self.world = world
    self.rank = rank
    self.dp_deg = self.mesh.dp_deg()
    self.dp_rank = dp_rank
    self.dp_groups = self.mesh.get_dp_groups_by_rank(self.rank)
    assert len(self.dp_groups) <= 1, "asymmetric TP inside a DP group is no longer supported"
    
    
    #sync_grad
    self.ordered_trainable_param = {}
    self.param_grad_ready = {}
    self.auto_perform_sync_grad = auto_perform_sync_grad

    self.grad_scale_ratio = 0.0
    self.sync_grad_hook_handles = [] 
    
    self.async_handles = []

    # setup
    self.set_local_batch_sizes(local_batch_sizes) # including register grad sync hooks
    self._build_ordered_trainable_param()
    self._broadcast_initial_param()
    
    self.prof.stop("DPE-__init__", True, False, self.enable_profiling)
    self.log(f"__init__ complete, dp_rank {self.dp_rank}, dp_groups {len(self.dp_groups)}")
    


  def forward(self, input, *args, **kwargs):
    assert not self.on_meta, "model must be instantiated at forward"
    
    self.log("forward")
    self.prof.start("DPE-forward", True, False, self.enable_profiling)
      
    self._sync_reqs()
    # self._sync_barrier()
    input = input.to(self.device)
    outputs = self.model(input, *args, **kwargs)
    
    self.log("forward complete")
    self.prof.stop("DPE-forward", True, False, self.enable_profiling)
      
    return outputs
  
  def instantiate(self, *args, skip_broadcast_initial_param: bool = False, **kwargs):
    self.log("instantiate")
    if self.on_meta:
      if hasattr(self.model, "instantiate"):
        self.model.instantiate()
      else:
        self.model = self.model.to_empty(device=self.device)
        reset_parameters(self)

      self._post_instantiate(*args, skip_broadcast_initial_param=skip_broadcast_initial_param, **kwargs)
    else:
      self.log("already instantiated")

  def _post_instantiate(self, *args, skip_broadcast_initial_param: bool = True, **kwargs):
    if not self.on_meta:
      return 
    
    self.log("_post_instantiate")
    if hasattr(self.model, "_post_instantiate"):
      self.model._post_instantiate(*args, **kwargs)
    instantiate_non_persistent_buffers(self, self.device)

    self.on_meta = False
    
    # special instantiate operation for DP (must be called after on_meta = False)
    if not skip_broadcast_initial_param:
      self._broadcast_initial_param()
    else:
      self.log("skip broadcast initial param")
    
    self._build_ordered_trainable_param()
    self._register_sync_grad_hooks()
    
  
  def get_disable_sync_func(self):
    def disable_sync_func():
      self.log("disable sync")
      self.enable_sync = False
    return disable_sync_func
  
  def get_enable_sync_func(self):
    def enable_sync_func():
      self.log("enable sync")
      self.enable_sync = True

    return enable_sync_func
  
  def set_local_batch_sizes(self, local_batch_sizes: Optional[List[int]] = None):
    self.log(f"set local batch sizes {local_batch_sizes}")

    if local_batch_sizes is None:
      local_batch_sizes = [1] * self.dp_deg
      
    global_micro_batch_size = sum(local_batch_sizes)
    self.grad_scale_ratio = local_batch_sizes[self.dp_rank] / global_micro_batch_size
    self._register_sync_grad_hooks()
  
  def log(self, msg: str, lvl: int = logging.DEBUG):
    if self.logger is not None:
      msg_wrapped = self.logging_pattern.format(msg)
      self.logger.log(level = lvl, msg = msg_wrapped)
  
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
  @PRIVATE
  """
  def _build_ordered_trainable_param(self):
    self.log("_build_ordered_trainable_param")
    if self.on_meta:
      self.log("model is on meta, skip _build_ordered_trainable_param")
      return
    
    self.ordered_trainable_param = {}
    named_params = list(self.named_parameters())
    named_params.sort(key=lambda x: x[0])

    for idx, (_, param) in enumerate(named_params):
      if not param.requires_grad:
        continue
      self.ordered_trainable_param[param] = idx
    self.ordered_trainable_param = OrderedDict(sorted(self.ordered_trainable_param.items(), key=lambda x: -x[1]))

  def _broadcast_initial_param(self):
    self.log("_broadcast_initial_param")

    if self.on_meta:
      self.log("model is on meta, skip _broadcast_initial_param")
      return
    
    with torch.no_grad():
      for name, param in self.model.named_parameters():
        # assert len(self.dp_groups) == 1 or hasattr(param, "slice_dim"), "param that belongs to multiple dp groups requires slice_dim attribute"
        
        if hasattr(param, "slice_dim") and isinstance(param.slice_dim, int): 
          handle = dist.broadcast(param, group_src = 0, group=self.dp_groups[0], async_op = True)
          self._track_reqs(handle)
        else:
          # not parallelizable layer
          # assume to be the same inside a TP group.
          # broadcast from first rank
            
          entire_dp_group = self.mesh.get_group()
          handle = dist.broadcast(param, group_src = 0, group=entire_dp_group, async_op=True)
          self._track_reqs(handle)
              
    self._sync_reqs()
    # self._sync_barrier()
    self.log("_broadcast_initial_param complete")

  def _perform_sync_grad(self):
    # remark: this will be called exactly ones every mini batch, and upon return, the allreduces are complete
    self.log("_perform_sync_grad")

    assert len(self.param_grad_ready) == len(self.ordered_trainable_param), "not all params are ready!"
    # to support partial activation graph without check, comment out the line above

      

    self.prof.start("_perform_sync_grad", True, False, self.enable_profiling)

    if self.dp_deg == 1:
      pass
    else:
      for param in self.ordered_trainable_param:
        if param.grad is None:
          param.grad = torch.zeros_like(param)
        if hasattr(param, "slice_dim") and isinstance(param.slice_dim, int): 
          group = self.dp_groups[0] # tp parallelizable
        else:
          group = self.mesh.get_group() # not tp parallelizable
        handle = dist.all_reduce(param.grad, group=group, async_op=True)
        self._track_reqs(handle)
        # handle = dist.all_reduce(param.grad, group=group, async_op=False)

    self.param_grad_ready = {}

    self._sync_reqs() # wait for all reduce to complete
    self.prof.start("_perform_sync_grad", True, False, self.enable_profiling)

    self.log("_perform_sync_grad complete")


  def _get_sync_grad_hook(self):
    
    def _sync_grad_hook(param: Tensor):
            
      if not self.enable_sync or not param.requires_grad:
        return

      # self.log(f"sync grad hook called and activated")
            
      # scale gradient by local batch size
      param.grad = param.grad * self.grad_scale_ratio

      # set ready
      self.param_grad_ready[param] = True
      if len(self.param_grad_ready) == len(self.ordered_trainable_param) and \
        self.auto_perform_sync_grad:
        self._perform_sync_grad()
    
    return _sync_grad_hook
  
  def _register_sync_grad_hooks(self):
    self.log("_register_sync_grad_hooks")
    if self.on_meta:
      self.log("model is on meta, skip _register_sync_grad_hooks")
      return

    # 1. must set param_grad_ready, so dp_deg == 1 doesn't mean no grad_sync
    # 2. some parameters may not be TP parallelizable, and grad must be averaged over entive DP-TP group.
    # if self.dp_deg == 1:
    #   return
    
    for handle in self.sync_grad_hook_handles:
      handle.remove()
    
    self.sync_grad_hook_handles = []  
    
    for param in self.model.parameters():
      if param.requires_grad:
        handle = param.register_post_accumulate_grad_hook(self._get_sync_grad_hook())
        self.sync_grad_hook_handles.append(handle)
  
  def _track_reqs(self, async_handle: Union[dist.Work, List[dist.Work], Callable]):
    if isinstance(async_handle, list):
      self.async_handles.extend(async_handle)
    elif isinstance(async_handle, dist.Work) or isinstance(async_handle, Callable):
      self.async_handles.append(async_handle)
    elif async_handle is None:
      pass
    else:
      assert False, f"unkown type of asyc_handle {type(async_handle)}"
  
  # wait for all pending communication to complete    
  def _sync_reqs(self):
    self.log(f"_sync_reqs {len(self.async_handles)}")
    if not self.async_handles:
      return
    
    self.prof.start("DPE-_sync_reqs", True, False, self.enable_profiling)
    
    for handle in self.async_handles:
      if isinstance(handle, dist.Work):
        handle.wait()
      else:
        handle()
    self.async_handles = []
    self.prof.stop("DPE-_sync_reqs", True, False, self.enable_profiling)
    
  def _sync_barrier(self):
    for dp_group in self.dp_groups:
      dist.barrier(group=dp_group)
  

# PPEngine is semi-transparent wrapper:
#   traning code modification required
#   no model code modification required

#   each call into forward will complete a mini batch, instead of micro batch, with loss function called
#   but optimizer and scheduler are leaved to user training code

class PPEngine(Module):
  
  #model
  model: PipelineExecModule
  
  #device
  device: torch.device
  mesh: DeviceMesh3D
  on_meta: bool
  
  #commu
  rank: int
  world: Optional[dist.ProcessGroup]
  
  #pipeline exec
  lbsz: int
  num_stage: int
  num_micro_batch: int
  pp_rank: int # stage_index
  dp_rank: int
  tp_rank: int
  
  schedules: List[PipelineExecOp]
  
  #logger
  logger: Optional[Logger]
  logging_pattern: str
  
  #profiler
  prof: Profiler
  enable_profiling: bool
  activation_mem_np_scale: Optional[float] # the ratio of non partitionable memory to output actvation memory
  
  def __init__(self, 
               model: Union[Module, List[Module], torch.nn.Sequential], # suggested to be on meta device
               device: torch.device,
               mesh: DeviceMesh3D, 
               lbsz_mesh: List[List[int]],
               strategy: str = "1F1B", # choice: ["GPipe", "1F1B"]
              #  num_stage: Optional[int] = None, # if leave to None, infer from mesh
               num_micro_batch: int = 1,
               batch_dim: int = 0,
               slice_dim: int = -1,
               tp_slice_dim: str = "column", # choice: ["column", "row"]
               init_on_meta: bool = False, # whether to initialize model on meta device, default to False.
               world: Optional[dist.ProcessGroup] = None,
               logger: Optional[Logger] = None,
               profiler: Optional[Profiler] = None, # if not provided, a new one will be created
               enable_profiling: bool = True
               ):
    super(PPEngine, self).__init__()
    
    # set device for NCCL for compatibility consideration (though it's encouraged to use CUDA_VISIBLE_DEVICES and cuda:0 instead, which is adpoted by suika cluster manager)
    device = torch.device(device)
    if device.type == "cuda":
      torch.cuda.set_device(device)

    rank = dist.get_rank(world)
    coord = mesh.get_coord(rank)
    pp_rank, dp_rank, tp_rank = coord
    
    #logger & profiler
    # moved to begining of __init__
    self.logger = logger
    self.logging_pattern = f"PPE[{rank}]@{coord}-{{}}"

    self.prof = Profiler(pattern = f"rank-{rank}-at-{coord}-{{}}", strategy="average_skip_first", device=device) \
                if profiler is None else \
                profiler
    self.enable_profiling = enable_profiling
        
    self.log("__init__")
    self.prof.start("PPE-__init__", True, False, self.enable_profiling)
    
    # model  
    if isinstance(model, Module):
      pass # assume partitioned model
    elif isinstance(model, list) or isinstance(model, torch.nn.Sequential):
      assert len(mesh) == len(model), f"number of stages mismatch between provided model({len(model)}) and device mesh({len(mesh)})"
      model = model[pp_rank]
    else:
      assert False, f"model type should be [List[Module], Sequential], but got {type(model)}"

    model = DPEngine(
                     model=model, 
                     device=device, 
                     mesh=mesh[pp_rank], 
                     local_batch_sizes=lbsz_mesh[pp_rank], 
                     auto_perform_sync_grad=False,
                     tp_slice_dim = tp_slice_dim, 
                     init_on_meta = init_on_meta,
                     world = world, 
                     logger = logger, 
                     profiler=self.prof, 
                     enable_profiling=enable_profiling)
 
    model = PipelineExecModule(     
                              model=model, 
                              device=device, 
                              mesh=mesh, 
                              lbsz_mesh=lbsz_mesh,
                              enable_grad_sync=model.get_enable_sync_func(),
                              disable_grad_sync=model.get_disable_sync_func(),
                              batch_dim=batch_dim,
                              slice_dim=slice_dim,
                              initial_grad_sync_is_enabled=False,
                              init_on_meta=init_on_meta,
                              world=world,
                              logger = logger,
                              profiler = self.prof,
                              enable_profiling = enable_profiling
                              ) # at this point, model is partitioned, so materialize it in place is acceptable

    self.model = model if init_on_meta else model.to(device)
    
    # device
    self.device = torch.device(device)
    self.mesh = mesh
    self.on_meta = init_on_meta
    
    # commu
    self.rank = rank
    self.world = world

    # pipeline exec
    self.lbsz = lbsz_mesh[pp_rank][dp_rank]
    self.num_stage = len(mesh)
    self.num_micro_batch = num_micro_batch
    self.pp_rank = pp_rank
    self.dp_rank = dp_rank
    self.tp_rank = tp_rank
    self.schedules = schedule_pipeline(strategy=strategy, num_stage=self.num_stage, num_micro_batch=self.num_micro_batch)

    self._init_world_commu()

    self.prof.stop("PPE-__init__", True, False, self.enable_profiling)
    self.log("__init__ complete")
    
  def forward(self, 
              inputs: Union[List[Any], Iterator[Any]], # list of mini-batch input tensors, or a dataloader
                            *args, 
              labels: Optional[Union[List[Any], Iterator[Any]]] = None, 
              loss_fn: Optional[Callable] = None, 
              require_outputs: Optional[bool] = None,
              require_loss: Optional[bool] = None,
              **kwargs):
    assert not self.on_meta, "model must be instantiated at forward"
    
    self.log("micro-batch forward")
    self.prof.start("PPE-forward", True, True, self.enable_profiling)
    
    assert (self.training and labels is not None and loss_fn is not None) or (not self.training and labels is None and loss_fn is None), f"training mode({self.training}) mismatch with labels({type(labels)}) and loss_fn({type(loss_fn)})"
    
    # set default "require"
    if require_loss is None and self.training: # 
      require_loss = True
    if require_outputs is None and not self.training: # default inference model
      require_outputs = True
    
    # buffer for return value
    outputs = [] if require_outputs and self.is_last_stage() else None
    loss_sum = 0 if require_loss and self.is_last_stage() else None
    
    # ignore unnecessary inputs/labels/loss_fn 
    if not self.is_first_stage():
      inputs = None
    if not self.is_last_stage():
      labels, loss_fn = None, None
              
    # helper function to extract mini-batch data from inputs/labels
    def extract_mini_batch(micro_batch: Union[List[Tensor], Iterator[Tensor]], mb_index: int):
      if micro_batch is None:
        return [] # as returned data need to be *args, so return [] instead of None
      self.prof.start(f"PPE-prepare-data", True, False, self.enable_profiling)
      if isinstance(micro_batch, List):
        extracted = micro_batch[mb_index]
      else:
        extracted = next(micro_batch)
      if not isinstance(extracted, List) and not isinstance(extracted, Tuple):
        extracted = [extracted]
      extracted = [entry.to(self.device) if isinstance(entry, Tensor) else entry for entry in extracted]
      self.prof.stop(f"PPE-prepare-data", True, False, self.enable_profiling)
      return extracted
    
    # exec schedules
    for clock_idx, exec_op in enumerate(self.schedules):
      
      if exec_op.skip(self.pp_rank): 
        # manual skip here is necessary, as we don't want to prepare input/label, or accumulate output/loss for skipped op
        # kind of design failure, the skip is intended to be handled by exec_op
        # the __call__ method for exec_op is nearly redundant, except for the grad_sync setting
        continue
      
      # if self.device.type == "cuda":
      #   self.prof.start(f"PPE-synchronize", True, False, self.enable_profiling)
      #   torch.cuda.synchronize(device=self.device)
      #   self.prof.stop(f"PPE-synchronize", True, False, self.enable_profiling)
      
      mb_index = exec_op.mb_index
      
      if exec_op.is_forward():
        self.log(f"forward mb={mb_index}")
        
        input = extract_mini_batch(inputs, mb_index)

        self.prof.start(f"PPE-forward-mb-{mb_index}", True, False, self.enable_profiling)
        output = exec_op(self.model, *input, *args, **kwargs)
        self.prof.stop(f"PPE-forward-mb-{mb_index}", True, False, self.enable_profiling)
        
        if outputs is not None:
          outputs.append(output)
        
      elif exec_op.is_backward():
        if not self.training: # in eval mode, skip backward
          continue
        self.log(f"backward mb={mb_index}")
        
        label = extract_mini_batch(labels, mb_index)
        
        self.prof.start(f"PPE-backward-mb-{mb_index}", True, False, self.enable_profiling)
        loss = exec_op(self.model, *label, loss_fn=loss_fn)
        self.prof.stop(f"PPE-backward-mb-{mb_index}", True, False, self.enable_profiling)
        
        if loss_sum is not None:
          loss_sum += loss.item()
      elif exec_op.is_sync():
        exec_op(self.model)
      else:
        assert False, f"invalid exec_op {exec_op}"
      
      # if self.device.type == "cuda":
      #   torch.cuda.synchronize(device=self.device)

    self.prof.stop("PPE-forward", True, True, self.enable_profiling)
    self.log("micro-batch forward complete")
    if self.training:
      return outputs, loss_sum # even if not required, a None will be returned for each
    else:
      return outputs
    
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
    
    self.log("_post_instantiate")
    if hasattr(self.model, "_post_instantiate"):
      self.model._post_instantiate(*args, **kwargs)
    instantiate_non_persistent_buffers(self, self.device)
    
    self.on_meta = False

  def is_last_stage(self) -> bool:
    return self.pp_rank == self.num_stage - 1

  def is_first_stage(self) -> bool:
    return self.pp_rank == 0
    
  def log(self, msg: str, lvl = logging.DEBUG):
    if self.logger is not None:
      msg_wrapped = self.logging_pattern.format(msg)
      self.logger.log(level = lvl, msg = msg_wrapped)
      
  def profile_result(self, require_time = True, require_memory = False, enable_profiling: Optional[bool] = None) -> Dict[str, Any]:
    # enable_profiling overrides the self.enable_profiling if provided
    if enable_profiling is None:
      enable_profiling = self.enable_profiling
    return self.prof.full_report(require_time, require_memory, enable_profiling)
  
  def profile_optimizer(self):
    @contextmanager
    def _profile():
        self.prof.start("optimizer", True, False, self.enable_profiling)
        try:
            yield
        finally:
            self.prof.stop("optimizer", True, False, self.enable_profiling)
    return _profile()
  
  def set_activation_mem_np_scale(self, scale: float):
    self.activation_mem_np_scale = scale

  def profile_result_to_scheduler(self, optimizer: Optional[torch.optim.Optimizer] = None) -> Union[Dict[str, float], None]:
    """
      compute constants required by estimator from given model.
      
      return fields are:
        *- iter_time:
          raw iteration time of a micro-batch -> to estimate accuracy only
        - param_raw: 
            size of (trainable) param in mb per tp, but not divided by num_layer
        - param_optim_mem_raw: 
            size of (all) param in mb per tp, but not divided by num_layer, including grad and optimizer state (for memory footprint)
        - activation_raw: 
            size of activation in mb *per layer* but not divided by lbsz, tp
        - activation_mem_raw: 
            size of activation in mb per tp, but not divied by num_layer, lbsz (for memory footprint)
        - forward_comp_raw: 
            raw time of forward of pure computation in seconds -> used to compute 'comp'
        *- forward_commu_raw: 
            raw time of forward communication in seconds -> to estimate accuracy only
        *- backward_comp_raw: 
            raw time of backward of pure computation in seconds -> to estimate accuracy only
        *- backward_commu_raw: 
            raw time of backward communication in seconds -> to estimate accuracy only
        - backward: 
            ratio of backward time to fwd time
        - backward_dp_commu_raw: 
            raw time of backward communication in seconds -> used to compute 'comp_commu_overlap'
        - optimizer_raw: (optional)
            optimizer latency per tp, but not divided by num_layer
        
      
      this should be called at the first stage of the pipeline for accuracy (rank 0)
    """
    if not self.enable_profiling or self.rank != 0:
      return None
      
    C = {} # constants to return

    def guarded_report(field, *args, **kwargs):
      result = self.prof.report(field, *args, **kwargs)
      if result is None:
        raise RuntimeError(f"guarded_report: {field}")
      else:
        return result
    
    try:
      iteration_field = "PPE-forward"
      iteration_time = guarded_report(iteration_field, True, False)
      if iteration_time is None:
        return None
      C["iter_time"] = iteration_time
      
      forward_raw_field = "PEM-forward-compute"
      # backward_raw_field = "PEM-backward-compute"
      # forward_raw_field = "PEM-forward-mb-0"
      backward_raw_field = "PEM-backward-mb-0"
      C["forward_raw_field"] = forward_raw_field
      C["backward_raw_field"] = backward_raw_field
      
      forward_raw = guarded_report(forward_raw_field, True, False)
      
      # param
      total_trainable_param_size = 0
      total_frozen_param_size = 0
      total_param_and_optimizer_size = 0
      
      for param in self.model.parameters():
        if param.requires_grad:
          total_trainable_param_size += param.numel() * param.element_size()
        else:
          total_frozen_param_size += param.numel() * param.element_size()
          
      if optimizer is not None:
        total_param_and_optimizer_size = total_trainable_param_size * 2 # including grad
        for state in optimizer.state.values():
          for k, v in state.items():
            if isinstance(v, torch.Tensor):
              total_param_and_optimizer_size += v.numel() * v.element_size()
        total_param_and_optimizer_size += total_frozen_param_size
      else:
        # assume Adam
        total_param_and_optimizer_size = total_trainable_param_size * 4 + total_frozen_param_size
      
      total_trainable_param_size /= 1024 * 1024 # convert to mb
      total_param_and_optimizer_size /= 1024 * 1024
      
      C["param_raw"] = total_trainable_param_size
      C["param_optim_mem_raw"] = total_param_and_optimizer_size
        
      # activation
      total_activation_size = 1
      for size in self.model.output_shape:
        total_activation_size *= size
      total_activation_size *= self.model.output_dtype.itemsize / (1024 * 1024) # convert to mb
      
      # remark: when TP is enabled, the activation mem size is slightly overshoot, as those activation not divided by tp is still counted as if it were.
      
      C["activation_raw"] = total_activation_size
      peak_memory_field = "PEM-forward-compute" # measured at first mb
      peak_memory = guarded_report(peak_memory_field, False, True)['peak']
      compensate_for_grad = self.num_micro_batch <= len(self.mesh) # n_mb <= pp_deg

      # observed_total_param_and_optimizer_size_wo_grad = torch.cuda.memory_allocated() / (1024 * 1024)
      total_param_and_optimizer_size_wo_grad = (total_param_and_optimizer_size - total_trainable_param_size)
      C["activation_mem_raw"] = peak_memory - total_param_and_optimizer_size_wo_grad - compensate_for_grad * total_trainable_param_size
      # C["activation_mem_raw"] = peak_memory - observed_total_param_and_optimizer_size_wo_grad - compensate_for_grad * total_trainable_param_size
      # self.log(f"? observed: {observed_total_param_and_optimizer_size_wo_grad}, estimated: {estimated_total_param_and_optimizer_size_wo_grad} memory")
      if self.activation_mem_np_scale is not None:
        C["activation_mem_np_scale"] = self.activation_mem_np_scale
      
      # forward_comp_raw
      def get_default(input, default):
        return input if input is not None else default
      def get_total_time(name):
        return get_default(self.prof.report(name), 0.0) * self.prof.count(name) # do not use guarded report here, some model may just not have related curtain commu fields
      def get_commu_time(commu_fields, count_field):
        if self.mesh[0][0].tp_deg() == 1:
          return 0
        count = self.prof.count(count_field)
        if count:
          return sum([get_total_time(field) for field in commu_fields]) / count
        else:
          return 0.0

      # forward
      CPL_fwd_commu_field = "CPL-forward-commu"
      RPL_fwd_commu_field = "RPL-forward-commu"
      forward_count_field = "PEM-forward" # num of micro-batch
      forward_commu_raw = get_commu_time([CPL_fwd_commu_field, RPL_fwd_commu_field], forward_count_field)
      forward_comp_raw = forward_raw - forward_commu_raw
      
      C["forward_comp_raw"] = forward_comp_raw
      C["forward_commu_raw"] = forward_commu_raw
      
      # backward
      backward_raw = guarded_report(backward_raw_field)

      # comp_commu_overlap
      backward_dp_commu_raw = guarded_report("PEM-grad-sync")
      if self.num_micro_batch == 1:
        backward_raw -= backward_dp_commu_raw
        
      CPL_bwd_commu_field = "CPL-backward-commu"
      RPL_bwd_commu_field = "RPL-backward-commu"
      ## backward commu is async, so its measurement is unreliable.
      backward_commu_raw = get_commu_time([CPL_bwd_commu_field, RPL_bwd_commu_field], forward_count_field)
      # backward_commu_raw = forward_commu_raw
      backward_comp_raw = backward_raw - backward_commu_raw
      
      backward = backward_comp_raw / forward_comp_raw
      if backward < 0: # only happen if serverly underutilize.
        self.log("failed to get profile: negative backward ratio", lvl=logging.ERROR)
        return None
      elif backward < 1:
        self.log("warned to get profile: backward ratio < 1; this usually indicates underutilization", lvl=logging.WARNING)
      
      C["backward"] = backward
      C["backward_comp_raw"] = backward_comp_raw
      C["backward_commu_raw"] = backward_commu_raw

      # dp communication time
      C["backward_dp_commu_raw"] = backward_dp_commu_raw

      # optimizer
      optimizer_raw = self.prof.report("optimizer")
      if optimizer_raw is None:
        self.log("warned to get profile: optimizer not profiled", lvl=logging.WARNING)
        optimizer_raw = 0.0

      C["optimizer_raw"] = optimizer_raw
      C["iter_time"] += optimizer_raw
        
    except Exception as e:

      self.log(f"failed to get profile result: {e}")
      if str(e).startswith("guarded_report"):
        return None
      else:
        raise e

    return C
  
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
  @PRIVATE
  """

  def _init_world_commu(self):
    # invoking batch-async-op without calling a collective primitive that all ranks participate first is not defined. so we need to perform a dummy communication, to setup the world.
    self.log("_init_world_commu")
    world_size = dist.get_world_size(group=self.world)
    if world_size <= 1:
      pass
    else:
      message = torch.tensor([1], device=self.device, dtype=torch.int)
      dist.all_reduce(message, op=dist.ReduceOp.SUM, group=self.world)
      assert message.item() == world_size, "world size mismatch"

      # # force p2p op buffer to be allocated (faster migration)
      # # deprecated, doing so slows down init too much
      # p2p_ops = []

      # send_tensor = torch.tensor([1] * world_size, device=self.device, dtype=torch.int)
      # recv_tensor = torch.zeros_like(send_tensor)
      # for src in range(world_size):
      #   for dst in range(world_size):
      #     if src == dst: continue
      #     if self.rank == src:
      #       self.log(f" {src}->{dst} ")
      #       p2p_ops.append(dist.P2POp(dist.isend, send_tensor[src], peer=dst, group=self.world))
      #     elif self.rank == dst:
      #       self.log(f" {src}->{dst} ")
      #       p2p_ops.append(dist.P2POp(dist.irecv, recv_tensor[src], peer=src, group=self.world))
      # reqs = maybe_batch_isend_irecv(p2p_ops)
      # self.log(f"_init_world_commu: p2p num of reqs : {len(reqs)}, num of ops {len(p2p_ops)}")
      # for req in reqs:
      #   req.wait()

      # assert recv_tensor.sum().item() == world_size - 1, f"world size mismatch expected {world_size - 1}, got {recv_tensor.sum().item()}"

    self.log("_init_world_commu complete")

  """
  @STATIC METHODS
  """
  @staticmethod
  def _partition_model_by_boundary_mark(model: Module):
    pass
  
"""
Alias for PPEngine
"""
SuikaModelEngine = PPEngine