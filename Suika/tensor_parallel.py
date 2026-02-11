from enum import Enum
import torch
from torch import Tensor
import torch.autograd
import torch.distributed as dist
from torch.nn import Module
from torch.nn import Parameter
from torch.nn.functional import linear

from typing import Callable, Dict, Iterable, Optional

from .device_mesh import DeviceMesh1D, DeviceMesh2D, DeviceMesh3D
# from .utils import is_meta_device
from .profiler import Profiler

"""
  ColumnWiseParallel
"""


def gather_along_dim(input_: Tensor, dim: int, rank: int = -1, world_size: int = -1, group: Optional[dist.ProcessGroup] = None) -> Tensor:
    """Gather alone a dimension."""
    if world_size == -1:
        world_size = dist.get_world_size(group=group)
    if world_size == 1:
      return input_
      
    output = [torch.zeros_like(input_) for _ in range(world_size)]
    dist.all_gather(output, input_, group=group)
    return torch.cat(output, dim=dim)
  
def split_along_dim(input_: Tensor, dim: int, rank: int = -1, world_size: int = -1, group: Optional[dist.ProcessGroup] = None) -> Tensor:
    """Split alone a dimension, no actually communication"""
    if world_size == -1:
        world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return input_
    
    if rank == -1:
        rank = dist.get_rank(group=group)
    
    return torch.chunk(input_, world_size, dim=dim)[rank]

class AllGatherFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_: Tensor, module: "ColumnParallelLinear") -> Tensor:
        """Forward function."""
        ctx.module = module
        return gather_along_dim(input_, -1, rank=module.rank, world_size=module.world_size, group=module.group)
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple:
        """Backward function."""
        module = ctx.module
        return split_along_dim(grad_output, -1, rank=module.rank, world_size=module.world_size, group=module.group), None
  
def tp_all_gather(input_: Tensor, module: "ColumnParallelLinear") -> Tensor:
    return AllGatherFunc.apply(input_, module)

class CopyFunc(torch.autograd.Function):
    # the f operator in Megatron-LM papaer
    
    @staticmethod
    def forward(ctx, input_: Tensor, module: "ColumnParallelLinear") -> Tensor:
        ctx.module = module
        return input_

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple:
        module = ctx.module
        
        module.prof.start("CPL-backward-commu", True, False, module.enable_profiling)
        if module.world_size > 1:
            dist.all_reduce(grad_output.contiguous(), group=module.group)
        module.prof.stop("CPL-backward-commu", True, False, module.enable_profiling)
        
        return grad_output, None
    
def tp_copy(input_: Tensor, module: "ColumnParallelLinear"):
    return CopyFunc.apply(input_, module)

class AsyncBackwardAllReduceLinear(torch.autograd.Function):
    # See LinearWithGradAccumulationAndAsyncCommunication in Megatron
    # overlap grad_input all_reduce with weight gradient computation
   
    @staticmethod
    def forward(ctx, input_: Tensor, weight: Tensor, bias: Optional[Tensor], module: "ColumnParallelLinear") -> Tensor:
        ctx.save_for_backward(input_, weight)
        ctx.bias = bias
        ctx.module = module
        
        output = torch.matmul(input_, weight.t())
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        input_, weight = ctx.saved_tensors
        bias: Optional[Tensor] = ctx.bias
        module: "ColumnParallelLinear" = ctx.module
        
        # compute grad_input
        grad_input = grad_output.matmul(weight) if input_.requires_grad else None
        
        # parallelize all_reduce with grad_weight computation
        handle = dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, group=module.group, async_op=True) \
            if grad_input is not None and module.world_size > 1 else None
                    
        input_features = weight.shape[1]
        output_features = weight.shape[0]
        
        grad_weight = grad_output.reshape(-1, output_features).t().matmul(input_.view(-1, input_features))
        grad_bias = grad_output.reshape(-1, output_features).sum(dim=0) if bias is not None else None
        
        module.prof.start("CPL-backward-commu", True, False, module.enable_profiling)
        if handle is not None:
            handle.wait()
        module.prof.stop("CPL-backward-commu", True, False, module.enable_profiling)
        
        return grad_input, grad_weight, grad_bias, None
   
   
def tp_async_backward_all_reduce_linear(input_: Tensor, weight: Tensor, bias: Optional[Tensor], module: "ColumnParallelLinear"):
    return AsyncBackwardAllReduceLinear.apply(input_, weight, bias, module)
    

class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism

    Args:
      input_features: First dimension of matrix A.
      output_features: Second dimension of matrix A.
      bias: If true, add bias.
      gather_output: If true, all gather output
      
    weight partitioned columnwise
    bias partitioned
    
    partitioned param has attr:
        slice_dim
        tp_deg
        tp_index
        tp_group
    """

    group: dist.ProcessGroup
    world_size: int
    # dist.get_world_size is pretty slow, so save world_size beforehand for performance
    
    input_features: int
    output_features: int
    gather_output: bool
    output_features_per_partition: int
    weight: Parameter
    bias: Optional[Parameter]
    prof: Profiler
    enable_profiling: bool
    
    default_async_backward_all_reduce = True

    def __init__(self, 
                 device: torch.device,
                 mesh: DeviceMesh1D,
                 input_features: int, 
                 output_features: int, 
                 bias: bool = True, 
                 gather_output: bool = True,
                 # overlap backward all_reduce with weight gradient computation
                 async_backward_all_reduce = None,
                 original_linear: Optional[torch.nn.Linear] = None,
                 init_on_meta: bool = False, 
                 profiler: Profiler = None,
                 enable_profiling: bool = False,
                ):
        super(ColumnParallelLinear, self).__init__()
        
        # commu
        self.group = mesh.get_tp_group()
        self.world_size = dist.get_world_size(group=self.group)
        self.rank = dist.get_rank(group=self.group)
        
        # linear
        self.input_features = input_features
        self.output_features = output_features
        self.gather_output = gather_output
        self.async_backward_all_reduce = async_backward_all_reduce if async_backward_all_reduce is not None else self.default_async_backward_all_reduce

        self.output_features_per_partition = output_features // len(mesh)

        if not init_on_meta:
            self.weight = Parameter(torch.empty(self.output_features_per_partition, input_features, device=device))

            if bias:
                self.bias = Parameter(torch.empty(self.output_features_per_partition, device=device))
            else:
                self.register_parameter('bias', None)
            
            if original_linear is not None:
                chunk = original_linear.weight.data.chunk(dim=0, chunks=len(mesh))[self.rank]
                self.weight.data.copy_(chunk)
                if self.bias is not None:
                    chunk = original_linear.bias.data.chunk(dim=0, chunks=len(mesh))[self.rank]
                    self.bias.data.copy_(chunk)
            else:
                self.reset_parameters()

        else:
            self.weight = Parameter(torch.empty(self.output_features_per_partition, input_features, device="meta"))
                
            if bias:
                self.bias = Parameter(torch.empty(self.output_features_per_partition, device="meta"))
            else:
                self.register_parameter('bias', None)
                
        # profiler
        self.prof = Profiler(f"rank-{dist.get_rank(group=self.group)}-{{}}", strategy="average_skip_first", device=device) if profiler is None else profiler
        self.enable_profiling = enable_profiling
        
        self._set_tp_attr()

    def forward(self, input_: Tensor) -> Tensor:        
        self.prof.start("CPL-forward", True, False, self.enable_profiling)
        
        if not self.async_backward_all_reduce:
            input_parallel = tp_copy(input_, self)
        else:
            input_parallel = input_
        
        # compute
        self.prof.start("CPL-forward-compute", True, False, self.enable_profiling)
        if self.async_backward_all_reduce:
            output_parallel = tp_async_backward_all_reduce_linear(input_parallel, self.weight, self.bias, self)
        else:
            output_parallel = linear(input_parallel, self.weight, self.bias)
        self.prof.stop("CPL-forward-compute", True, False, self.enable_profiling)
        
        # communicate
        self.prof.start("CPL-forward-commu", True, False, self.enable_profiling) 
        if self.gather_output:
            output = tp_all_gather(output_parallel, self)
        else:
            output = output_parallel
        self.prof.stop("CPL-forward-commu", True, False, self.enable_profiling)

        self.prof.stop("CPL-forward", True, False, self.enable_profiling)

        return output
    
    def __str__(self):
        return f"ColumnParallelLinear(input_features={self.input_features}, output_features={self.output_features}, bias={self.bias is not None}, gather_output={self.gather_output})"
    def __repr__(self):
        return self.__str__()
    
    def _set_tp_attr(self):
        params = [self.weight, self.bias] if self.bias is not None else [self.weight]
        for param in params:
            param.__setattr__("slice_dim", 0) # given that weight is actually transposed, the slice_dim is 0 rather than -1
            param.__setattr__("tp_deg", self.world_size)
            param.__setattr__("tp_index", self.rank)
            param.__setattr__("tp_group", self.group)

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=torch.sqrt(torch.tensor(5)).item())
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
"""
  RowWiseParallel
"""

class AllReduceFunc(torch.autograd.Function):
    # the g operator in Megatron-LM papaer
    
    @staticmethod
    def forward(ctx, input_: Tensor, module: "RowParallelLinear") -> Tensor:
        ctx.module = module
        
        if module.world_size == 1:
            return input_
        dist.all_reduce(input_.contiguous(), group=module.group)
        return input_

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple:
        return grad_output, None

def tp_all_reduce(input: Tensor, module: "RowParallelLinear") -> Tensor:
    return AllReduceFunc.apply(input, module)

class ScatterFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: Tensor, module: "RowParallelLinear") -> Tensor:
        ctx.module = module
        if module.world_size == 1:
            return input_
        
        return split_along_dim(input_, -1, rank=module.rank, world_size=module.world_size, group=module.group)
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple:
        module = ctx.module
        module.prof.start("RPL-backward-commu", True, False, module.enable_profiling)
        grad_input = gather_along_dim(grad_output, -1, rank=module.rank, world_size=module.world_size, group=module.group)
        module.prof.stop("RPL-backward-commu", True, False, module.enable_profiling)
        
        return grad_input, None
    
def tp_scatter(input: Tensor, module: "RowParallelLinear") -> Tensor:
    return ScatterFunc.apply(input, module)

class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism

    Args:
      input_features: First dimension of matrix A.
      output_features: Second dimension of matrix A.
      bias: If true, add bias.
      input_is_parallel: If true, assume input is already split across GPUs.
      
      weight partitioned rowsise
      bias not partitioneded
    
      partitioned param has attr:
        slice_dim
        tp_deg
        tp_index
        tp_group
    """
    
    group: dist.ProcessGroup
    world_size: int
    rank: int
    
    input_features: int
    output_features: int
    input_is_parallel: bool
    input_features_per_partition: int
    weight: Parameter
    bias: Optional[Parameter]
    prof: Profiler
    enable_profiling: bool

    def __init__(
        self, 
        device: torch.device,
        mesh: DeviceMesh1D,
        input_features: int, 
        output_features: int, 
        bias: bool = True, 
        input_is_parallel: bool = False, 
        original_linear: Optional[torch.nn.Linear] = None,
        init_on_meta: bool = False,
        profiler: Optional[Profiler] = None,
        enable_profiling: bool = False
    ):
        super(RowParallelLinear, self).__init__()
        
        # commu
        self.group = mesh.get_tp_group()
        self.world_size = dist.get_world_size(group=self.group)
        self.rank = dist.get_rank(group=self.group)
        
        # linear
        self.input_features = input_features
        self.output_features = output_features
        self.input_is_parallel = input_is_parallel

        self.input_features_per_partition = input_features // len(mesh)

        if not init_on_meta:
            self.weight = Parameter(torch.empty(output_features, self.input_features_per_partition, device=device))
            if bias:
                self.bias = Parameter(torch.empty(output_features, device=device))
            else:
                self.register_parameter('bias', None)

            if original_linear is not None:
                chunk = original_linear.weight.data.chunk(dim=-1, chunks=len(mesh))[self.rank]
                self.weight.data.copy_(chunk)
                if self.bias is not None:
                    self.bias.data.copy_(original_linear.bias.data)
            else:
                self.reset_parameters()
        else:
            self.weight = Parameter(torch.empty(output_features, self.input_features_per_partition, device="meta"))
                
            if bias:
                self.bias = Parameter(torch.empty(output_features, device="meta"))
            else:
                self.register_parameter('bias', None)
        
            
        # profiler 
        self.prof = Profiler(f"rank-{dist.get_rank(group=self.group)}-{{}}", strategy="average_skip_first", device=device) if profiler is None else profiler
        self.enable_profiling = enable_profiling

        self._set_tp_attr()

    def forward(self, input_: Tensor) -> Tensor:
        self.prof.start("RPL-forward", True, False, self.enable_profiling)
        
        # scatter input if necessary
        # do not count this into commu
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = tp_scatter(input_, self)
    
        # compute
        self.prof.start("RPL-forward-compute", True, False, self.enable_profiling)
        output_parallel = linear(input_parallel, self.weight, None)
        # bias cannot be applied here, since the all reduce sums over all bias, which applies bias to world_size times
        self.prof.stop("RPL-forward-compute", True, False, self.enable_profiling)
        
        # communicate output
        self.prof.start("RPL-forward-commu", True, False, self.enable_profiling)
        if self.world_size > 1:
            output = tp_all_reduce(output_parallel, self)
        else:
            output = output_parallel
        self.prof.stop("RPL-forward-commu", True, False, self.enable_profiling)
        
        # apply bias
        # do not count this into comp
        output = output + self.bias if self.bias is not None else output
        
        self.prof.stop("RPL-forward", True, False, self.enable_profiling)
        return output
    
    def __str__(self):
        return f"RowParallelLinear(input_features={self.input_features}, output_features={self.output_features}, bias={self.bias is not None}, input_is_parallel={self.input_is_parallel})"
    def __repr__(self):
        return self.__str__()

    def _set_tp_attr(self):
        param = self.weight
        if param.requires_grad:
            param.__setattr__("slice_dim", -1) # given that weight is actually transposed, the slice_dim is -1 rather than 0
            param.__setattr__("tp_deg", self.world_size)
            param.__setattr__("tp_index", self.rank)
            param.__setattr__("tp_group", self.group)

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=torch.sqrt(torch.tensor(5)).item())
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

"""
  AutoParallizer
  
  Customize parallel strategy can be specified by setting "tp_meta" attribute to module.
  
  tp_meta fields are:
    - slice_dim: either column or row 
    - kwargs: {
        "gather_output" : bool (optional, only valid with column)
        "input_is_parallel" : bool (optional, only valid with row)
    }    

  the setting applies to all children of the module

  Another useful tp attribute to module is:
    - div_by_tp_deg: Union[str, List[str]] 
    (optional, divide the given attribute of module by tp_deg)
    example usage is to change num_heads in GPTBlock
    
    
  TPMeta enum class provides all possible template for tp_meta
"""

class TPMeta():
    COLUMN_NO_GATHER = \
        { "slice_dim": "column" , "kwargs" : {"gather_output": False }}
    COLUMN_GATHER = \
        { "slice_dim": "column" , "kwargs" : {"gather_output": True }}
    ROW_NO_SCATTER = \
        { "slice_dim": "row" , "kwargs" : {"input_is_parallel": True }}
    ROW_SCATTER = \
        { "slice_dim": "row" , "kwargs" : {"input_is_parallel": False }}
        
    def __getitem__(self, key):
        return self.value[key]

# replace linear layer with TP parallel linear based on slice_dim hint, also set attribute slice_dim to param tensor
# note: modification to model is in place
def parallelize_tp(
                    model: Module,
                    device: torch.device,
                    mesh: DeviceMesh1D,
                    slice_dim: str, # choice: "column"/"row"/"none"
                    init_on_meta: bool = False,
                    profiler: Optional[Profiler] = None,
                    enable_profiling: bool = False,
                    ) -> Module:
    # this is kind of hacky
    # we need slice dim set even if its TP deg is 1
    # example setup:
    #  [0], [1, 2] in same DP group, requires 0 to have slice_dim set
    
    # if len(mesh) == 1:
    #     return model
    
    assert isinstance(model, torch.nn.Module), f"model is {type(model)} and should be torch.nn.Module"
    
    if slice_dim == "none": # remember to set slice_dim/tp_deg/tp_group attribute by yourself !!!
        return model if init_on_meta else model.to(device)
    
    def slice_at_column(module: Module, tp_meta: Dict):
        # Extract general kwargs required by the constructor
        general_kwargs = {k: v for k, v in tp_meta["kwargs"].items() if k in ["device", "mesh", "init_on_meta", "profiler", "enable_profiling"]}
        # Extract specific kwargs for ColumnParallelLinear from tp_meta["kwargs"], providing defaults matching the constructor
        specific_kwargs = {
            "gather_output": tp_meta["kwargs"].get("gather_output", True),
            "async_backward_all_reduce": tp_meta["kwargs"].get("async_backward_all_reduce", None)
        }

        new_module = ColumnParallelLinear(
            input_features=module.in_features,
            output_features=module.out_features,
            bias=module.bias is not None,
            original_linear=module,
            **general_kwargs, # Pass general args like device, mesh, etc.
            **specific_kwargs # Pass specific args like gather_output
        )
        return new_module
    
    def slice_at_row(module: Module, tp_meta: Dict):
        # Extract general kwargs required by the constructor
        general_kwargs = {k: v for k, v in tp_meta["kwargs"].items() if k in ["device", "mesh", "init_on_meta", "profiler", "enable_profiling"]}
        # Extract specific kwargs for RowParallelLinear from tp_meta["kwargs"], providing defaults matching the constructor
        specific_kwargs = {
            "input_is_parallel": tp_meta["kwargs"].get("input_is_parallel", False)
        }
        
        new_module = RowParallelLinear(
            input_features=module.in_features, 
            output_features=module.out_features, 
            bias=module.bias is not None,
            original_linear=module,
            **general_kwargs, # Pass general args like device, mesh, etc.
            **specific_kwargs # Pass specific args like input_is_parallel
        )
        return new_module
    
    def slice_set_attr(module: Module):
        # SOLVED : do nothing, DPEngine will handle it.
        #          do nothing here
        
        ## the set logic is handled at __init__ now        
        return module
    
    
    def recursive_parallelize(module: Module, tp_meta: Dict) -> Module:
        
        current_tp_meta = tp_meta.copy() # Create a copy to avoid modifying the dict for siblings/parent
        current_tp_meta["kwargs"] = tp_meta["kwargs"].copy() # Deep copy kwargs as well

        if hasattr(module, "tp_meta"): # respect custom tp_meta
            current_tp_meta["slice_dim"] = module.tp_meta["slice_dim"]
            
            # Update kwargs: Start with parent's kwargs, then overlay module's specific kwargs
            current_tp_meta["kwargs"].update(module.tp_meta.get("kwargs", {}))
            
        
        if hasattr(module, "div_by_tp_deg"): # respect custom div_by_tp_deg
            attrs = getattr(module, "div_by_tp_deg")
            if not isinstance(attrs, Iterable):
                attrs = [attrs]
            for attr in attrs:
                if hasattr(module, attr):
                    old_attr = getattr(module, attr)
                    tp_deg = len(mesh)
                    assert old_attr % tp_deg == 0, f"Attribute {attr} is not divisible by tp_deg {tp_deg}"
                    setattr(module, attr, old_attr // tp_deg)
                else:
                    raise ValueError(f"Attribute {attr} not found in {module.__class__.__name__}")
        
        if isinstance(module, torch.nn.Linear):            
            slice_dim = current_tp_meta["slice_dim"]
            
            if slice_dim == "column":
                return slice_at_column(module, current_tp_meta)
            elif slice_dim == "row":
                return slice_at_row(module, current_tp_meta)
            else:
                assert False, f"Invalid slice_dim: {slice_dim}"

        for name, child in list(module.named_children()):
            new_child = recursive_parallelize(child, current_tp_meta)
            setattr(module, name, new_child)

        return module

    tp_meta = { 
               "slice_dim": slice_dim, 
               "kwargs": {
                   "device": device, 
                   "mesh": mesh, 
                   "init_on_meta": init_on_meta, 
                   "profiler": profiler, 
                   "enable_profiling": enable_profiling
                } 
               }
    model = recursive_parallelize(model, tp_meta)
    
    return model if init_on_meta else model.to(device)









"""UTILS"""

def set_tp_attr(model: torch.nn.Module):
  if isinstance(model, ColumnParallelLinear) or isinstance(model, RowParallelLinear):
    model._set_tp_attr()
  else:
    for name, child in model.named_children():
      set_tp_attr(child)

def check_slice_dim_lost(model: torch.nn.Module):
  if isinstance(model, ColumnParallelLinear) or isinstance(model, RowParallelLinear):
    assert hasattr(model.weight, "slice_dim"), \
      f"slice_dim is not found in {model.__class__.__name__}"
  for name, child in model.named_children():
    check_slice_dim_lost(child)