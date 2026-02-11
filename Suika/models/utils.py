from typing import Iterable, Tuple, Union, TYPE_CHECKING
import copy
import torch
from torch import nn
from torch.nn import Module
from transformers import PretrainedConfig as HuggingfaceConfig

if TYPE_CHECKING:
  from Suika.external.Rubick.utils import PerfParams as RubickPerfParams
  from Suika.external.Sia.utils import PerfParams as SiaPerfParams
  
"""
An abstract class representing a transformer model configuration.
Primiarily for interface with Metis and Rubick.
"""
class ModelConfig():
  
  # basic model configuration
  
  model_name: str
  # ... these variables' names differ between models. we need unified representation
  n_layer: int # aka. num hidden layers
  n_embd: int # aka. hidden size
  n_positions: int # aka. seq len
  n_attn_head: int
  vocab_size: int 

  # estimator constants, for both simulation and Rubick memory assessment
  estimator_C: dict 
  
  # performance parameters, for Rubick
  rubick_perf_params: "RubickPerfParams"
  atomic_forward_time: float # time of the forward pass per sample per iteration
  total_param_size: float # total parameter size in MB
    
  # performance parameters, for Sia
  sia_perf_params: "SiaPerfParams"
    
  # native huggingface config, for model creation
  native_config: HuggingfaceConfig
  
  def __init__(self, 
                model_name: str, 
                n_layer: int, 
                n_embd: int, 
                n_positions: int,
                n_attn_head: int = None,
                vocab_size: int = None,
                estimator_C: dict = None,
                rubick_perf_params: "RubickPerfParams" = None,
                atomic_forward_time: float = None,
                total_param_size: float = None,
                sia_perf_params: "SiaPerfParams" = None,
                native_config: HuggingfaceConfig = None
              ):
    self.model_name = model_name
    self.n_layer = n_layer
    self.n_embd = n_embd
    self.n_positions = n_positions
    self.n_attn_head = n_attn_head
    self.vocab_size = vocab_size
    
    self.estimator_C = estimator_C

    if isinstance(rubick_perf_params, dict):
      from Suika.external.Rubick.utils import PerfParams as RubickPerfParams
      rubick_perf_params = PerfParams(**rubick_perf_params)
    
    if isinstance(rubick_perf_params, dict):
      from Suika.external.Sia.utils import PerfParams as SiaPerfParams
      sia_perf_params = PerfParams(**sia_perf_params)
    
      
    self.rubick_perf_params = rubick_perf_params
    self.atom_forward_time = atomic_forward_time
    self.total_param_size = total_param_size
  
    self.sia_perf_params = sia_perf_params
  
    self.set_estimator_C(estimator_C)

    self.native_config = native_config
    
  def set_estimator_C(self, estimator_C: dict, override: bool = False):
    """
    Set the estimator constants for the model.
    """
    self.estimator_C = estimator_C

    if estimator_C is not None:
      from Suika.external.Rubick.hyperparam import k_acc_TFLOPS
      if self.atom_forward_time is None or override:
        from Suika.external.Rubick.hyperparam import k_acc_TFLOPS
        self.atom_forward_time = estimator_C["comp"] / k_acc_TFLOPS
      if self.total_param_size is None or override:
        self.total_param_size = estimator_C["param"] * self.n_layer    
  
  def get_max_tp_allowed(self):
    from Suika.cluster.resource import default_n_acc_per_node
    max_tp_allowed = default_n_acc_per_node
    while self.n_attn_head % max_tp_allowed != 0:
      max_tp_allowed /= 2
    return int(max_tp_allowed)
  
  """
  Create a ModelConfig from a huggingface model name and config.
  """
  @staticmethod
  def from_huggingface(model_name: str, config: HuggingfaceConfig, **kwargs) -> "ModelConfig":
    # infer n_layer, n_embd, n_positions from huggingface config
    # native_config will be set
    
    from Suika.models.gpt_model import GPT2Config
    from Suika.models.llama_model import LlamaConfig
    from Suika.models.qwen_model import QwenConfig
    if isinstance(config, GPT2Config):
      return ModelConfig(
        model_name=model_name,
        n_layer=config.n_layer,
        n_embd=config.n_embd,
        n_positions=config.n_positions,
        n_attn_head=config.n_head,
        vocab_size=config.vocab_size,
        native_config=config,
        **kwargs
      )
    if isinstance(config, LlamaConfig):
      return ModelConfig(
        model_name=model_name,
        n_layer=config.num_hidden_layers,
        n_embd=config.hidden_size,
        n_attn_head=min(config.num_attention_heads, config.num_key_value_heads),
        n_positions=config.max_position_embeddings,
        vocab_size=config.vocab_size,
        native_config=config,
        **kwargs
      )
    if isinstance(config, QwenConfig):
      return ModelConfig(
        model_name=model_name,
        n_layer=config.num_hidden_layers,
        n_embd=config.hidden_size,
        n_attn_head=min(config.num_attention_heads, config.num_key_value_heads),
        n_positions=config.max_position_embeddings,
        vocab_size=config.vocab_size,
        native_config=config,
        **kwargs
      )
    raise ValueError(f"Unknown model type {model_name}.")
  
  """
  Create a ModelConfig from predefined specifications
  """
  @staticmethod
  def from_spec(model_spec: str) -> "ModelConfig":
    from Suika.models.gpt_model import ModelConfig_GPT_specs
    from Suika.models.llama_model import ModelConfig_Llama_specs
    from Suika.models.qwen_model import ModelConfig_Qwen_specs
    
    try:
      if "GPT" in model_spec:
        return copy.deepcopy(ModelConfig_GPT_specs[model_spec])
      if "Llama" in model_spec:
        return copy.deepcopy(ModelConfig_Llama_specs[model_spec])
      if "Qwen" in model_spec:
        return copy.deepcopy(ModelConfig_Qwen_specs[model_spec])
      raise ValueError(f"Unkown model type {model_spec}.")
    except KeyError:
      raise ValueError(f"Model {model_spec} not found in ModelConfig specs.")
  
  """
  Get the model_provider from the native config, that can be used with PlanConfig.to_PPEngine_kwargs.
  
  Return:
    Callable[[int, int], Module]: model_provider(layer_index, num_layer) -> model
  """
  def get_model_provider(self, **kwargs):
    if self.native_config is None:
      raise ValueError("native_config is not set. Cannot get model provider.")
    
    from Suika.models.gpt_model import get_model_provider_GPT, GPTConfig
    from Suika.models.llama_model import get_model_provider_Llama, LlamaConfig
    from Suika.models.qwen_model import get_model_provider_Qwen, QwenConfig
    
    if isinstance(self.native_config, GPTConfig):
      return get_model_provider_GPT(config=self.native_config, **kwargs)

    if isinstance(self.native_config, LlamaConfig):
      return get_model_provider_Llama(config=self.native_config, **kwargs)
    
    if isinstance(self.native_config, QwenConfig):
      return get_model_provider_Qwen(config=self.native_config, **kwargs)
    
    raise ValueError(f"Unknown model type {self.model_name}.")

  @staticmethod
  def get_model_provider_from_spec(model_spec: str, **kwargs):
    return ModelConfig.from_spec(model_spec).get_model_provider(**kwargs)
  
  
  def __str__(self):
    return f"ModelConfig({self.model_name}, n_layer={self.n_layer}, n_embd={self.n_embd}, n_positions={self.n_positions}, vocab_size={self.vocab_size}, estimator_C={self.estimator_C is not None}, rubick={self.rubick_perf_params is not None}, sia={self.sia_perf_params is not None}, native_config={self.native_config is not None})"
  
  def __repr__(self):
    return self.__str__()


"""
ModelStrategy defines a user task with:
1. model config
2. execution plan (of an uniform plan config, and n_iter)
"""
class ModelStrategy():
  model_config: ModelConfig
  acc_type: str
  global_batch_size: int
  num_micro_batch: int
  pp_dp_tp: Tuple[int, int, int] # pp, dp, tp
  
  n_iter: int # number of iterations for the task
  iter_time: float # time of estimated iteration time (if available)

  def __init__(
    self,
    model_config: ModelConfig,
    acc_type: str,
    global_batch_size: int,
    num_micro_batch: int,
    pp_dp_tp: Tuple[int, int, int],
    n_iter: int = None,
    iter_time: float = None,
  ):
    self.model_config = model_config
    self.acc_type = acc_type
    self.pp_dp_tp = pp_dp_tp
    self.global_batch_size = global_batch_size
    self.num_micro_batch = num_micro_batch
    
    self.n_iter = n_iter
    self.iter_time = iter_time
    
    if not self.validate():
      return None
    
  def validate(self) -> bool:
    pp, dp, tp = self.pp_dp_tp
    if self.global_batch_size % (dp * self.num_micro_batch) != 0:
      return False
    # if self.model_config.n_layer % pp != 0:
    #   return False
    return True
    
  def get_plan_config(self, **kwargs):
    from Suika.cluster.plan_config import get_uniform_plan_config

    
    kwargs_ = {
      "acc_type": self.acc_type,
      "pp_dp_tp": self.pp_dp_tp,
      "n_layer": self.model_config.n_layer,
      "gbs": self.global_batch_size,
      "n_mb": self.num_micro_batch,
      "max_tp_allowed": self.model_config.get_max_tp_allowed(),
    }
    kwargs_.update(kwargs)
    return get_uniform_plan_config(
      **kwargs_
    )
  
  def get_model_provider(self, **kwargs):
    return self.model_config.get_model_provider(**kwargs)
  
  def get_world_size(self):
    pp, dp, tp = self.pp_dp_tp
    return pp * dp * tp
  
  @staticmethod
  def from_spec(model_spec: str):
    from Suika.models.gpt_model import ModelStrategy_GPT_specs
    from Suika.models.llama_model import ModelStrategy_Llama_specs
    from Suika.models.qwen_model import ModelStrategy_Qwen_specs
    
    try:
      if "GPT" in model_spec:
        return copy.deepcopy(ModelStrategy_GPT_specs[model_spec])
      if "Llama" in model_spec:
        return copy.deepcopy(ModelStrategy_Llama_specs[model_spec])
      if "Qwen" in model_spec:
        return copy.deepcopy(ModelStrategy_Qwen_specs[model_spec])
      raise ValueError(f"Unknown model type {model_spec}.")
    except KeyError:
      raise ValueError(f"Model {model_spec} not found in ModelStrategy specs.")

  def __str__(self):
    return f"ModelStrategy(\n" \
           f"  {self.model_config},\n" \
           f"  acc_type={self.acc_type}, global_batch_size={self.global_batch_size}, num_micro_batch={self.num_micro_batch}, pp_dp_tp={self.pp_dp_tp}, n_iter={self.n_iter}\n" \
           f")"
  def __repr__(self):
    return self.__str__()
  
"""
@EXPORT
Get model's default plan_config from predefined model spec.
"""
def get_plan_config_from_spec(model_spec: str, **kwargs):
  return ModelStrategy.from_spec(model_spec).get_plan_config(**kwargs)

"""
@EXPORT
Get model provider function from predefined model spec.
Return:
  Callable[[int, int], Module]: model_provider(layer_index, num_layer) -> model
"""
def get_model_provider_from_spec(model_spec: str, **kwargs):
  return ModelStrategy.from_spec(model_spec).get_model_provider(**kwargs)

"""
@EXPORT
Get model's estimator_C from predefined model spec.
"""
def get_estimator_C_from_spec(model_spec: str):
  return ModelConfig.from_spec(model_spec).estimator_C


"""
  RenameModule is a wrapper around a PyTorch module that assigns the model to the attribute of given name.
  
  This can be useful for adding a prefix to the state_dict keys, 
  distinguishing layers globally in a pipeline setup.
  
  Attributes:
    name (str): The name assigned to the wrapped module.
    model (Module): The wrapped PyTorch module.

  Methods:
    forward(*args, **kwargs): Forwards the input arguments to the wrapped module's forward method.
    get_name(): Returns the name of the wrapped module.
    get_model(): Returns the wrapped module.
    __str__(): Returns the string representation of the wrapped module.
    __repr__(): Returns the representation of the wrapped module.
"""
class RenameModule(Module):
    
  name : str
  
  def __init__(self, model: Module, name: str):
    super().__init__()
    setattr(self, str(name), model)
    self.name = name
    
  def forward(self, *args, **kwargs):
    return self.get_model()(*args, **kwargs)
    
  def get_name(self):
    return self.name
  
  def get_model(self):
    return getattr(self, self.name)
    
  def __str__(self):
    return f"[{self.name}] " + str(self.get_model())
  
  def __repr__(self):
    return f"[{self.name}] " + repr(self.get_model())
  
  
"""
  OffsetModuleList is reimplement of ModuleList that allows for an offset in the indices of the modules.
  
  Mostly for reloading state_dict from a checkpoint.
  
  (attribute is reindexed, but subscript access is still 0 indexed)
"""
class OffsetModuleList(Module):
  offset: int
  
  def __init__(self, modules: Iterable[Module], offset: int = 0):
    super(OffsetModuleList, self).__init__()
    self.offset = offset

    for i, module in enumerate(modules):
      self.add_module(str(i + self.offset), module)

  def __getitem__(self, idx: Union[int, slice]) -> Union[Module, "OffsetModuleList"]:
    if isinstance(idx, slice):
      new_offset = self.offset + (idx.start if idx.start >= 0 else idx.start + len(self))
      return self.__class__([self.__getitem__(i) for i in range(*idx.indices(len(self)))], new_offset)
    
    if idx < 0:
      idx += len(self)
    key = str(idx + self.offset)
    return self._modules[key]

  def __len__(self) -> int:
    return len(self._modules)   

  def append(self, module: Module) -> None:
    key = str(self.offset + len(self))
    self.add_module(key, module)

  def extend(self, modules: Iterable[Module]) -> "OffsetModuleList":
    for module in modules:
      self.append(module)
    return self

  def __iter__(self) -> Iterable[Module]:
    return iter(self._modules.values())
  
  
"""
@EXPORT
Replace shallowest ModuleList in the module with OffsetModuleList
Expect: only one module list wrapps all the block
"""
def replace_ModuleList_with_OffsetModuleList(module: Module, offset: int = 0) -> Module:
  if isinstance(module, nn.ModuleList):
    return OffsetModuleList(module, offset)
  else:
    for name, submodule in module.named_children():
      setattr(module, name, replace_ModuleList_with_OffsetModuleList(submodule, offset))
  return module

"""
@EXPORT

replace this with transformer implementation Conv1D to support TP
"""
from Suika.external.Rubick.utils import PerfParams
from Suika.tensor_parallel import ColumnParallelLinear
from transformers.pytorch_utils import Conv1D as HuggingfaceConv1D

class SuikaConv1D(nn.Module):
    """
    reuse nn.Linear so that auto parallelize_tp works
    
    also divides nf by tp_deg for TP compatibility

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nx: int, nf: int, init_on_meta: bool = False):
      self.nf = nf
      super().__init__()
      self.linear = nn.Linear(nx, nf, device="meta" if init_on_meta else None)
      nn.init.normal_(self.linear.weight, std=0.02)

    def forward(self, x):
      nf = self.nf
      if isinstance(self.linear, ColumnParallelLinear) and not self.linear.gather_output:
      # support parallel output
        nf //= self.linear.world_size
      size_out = x.size()[:-1] + (nf,)

      x = self.linear(x)
      x = x.view(size_out)
      return x
    
def replace_Conv1D(module: nn.Module, init_on_meta: bool, copy_param: bool):
  if isinstance(module, HuggingfaceConv1D):
    # replace Conv1D with SuikaConv1D
    (nx, nf) = module.weight.data.size()
    new_module = SuikaConv1D(nx, nf, init_on_meta = init_on_meta)
    if copy_param and not init_on_meta:
      with torch.no_grad():
        new_module.linear.weight.copy_(module.weight.data.transpose(0, 1))
        new_module.linear.bias.copy_(module.bias.data)
    return new_module
  else:
    for name, child in module.named_children():
      new_child = replace_Conv1D(child, init_on_meta, copy_param)
      setattr(module, name, new_child)
    return module
  
"""
Compute model parameters for a given model

Return:
  int: 
  if output_bytes is False: (default)
    number of parameters in the model
  else:
    size of the model in bytes
"""
def compute_model_parameter_size(model: Module, output_bytes: bool = False) -> int:
  total_params = 0
  for param in model.parameters():
    if output_bytes:
      total_params += param.numel() * param.element_size()
    else:
      total_params += param.numel()
  return total_params