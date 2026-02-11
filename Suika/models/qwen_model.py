from Suika.models.utils import (
  ModelConfig, ModelStrategy,
  replace_ModuleList_with_OffsetModuleList,
)

from Suika.tensor_parallel import TPMeta
from Suika.utils import set_default_device_context

from typing import Callable, Dict, Optional, Tuple
import torch
import torch.nn as nn

from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2DecoderLayer, Qwen2RotaryEmbedding, Qwen2RMSNorm

QwenConfig = Qwen2Config

def adopt_Megatron_TP_for_Qwen(h: nn.ModuleList):
  """
  Adapts QwenDecoderLayers for Megatron-style Tensor Parallelism.
  Sets tp_meta attributes on relevant nn.Linear modules within QwenAttention and QwenMLP.
  """
  for block in h: # block is QwenDecoderLayer
    # Self Attention (QwenAttention)
    if hasattr(block, 'self_attn'):
      block.self_attn.q_proj.tp_meta = TPMeta.COLUMN_NO_GATHER
      block.self_attn.k_proj.tp_meta = TPMeta.COLUMN_NO_GATHER
      block.self_attn.v_proj.tp_meta = TPMeta.COLUMN_NO_GATHER
      block.self_attn.o_proj.tp_meta = TPMeta.ROW_NO_SCATTER
      # Attributes to be divided by TP degree
      block.self_attn.div_by_tp_deg = ["num_heads", "num_key_value_heads", "hidden_size"]
      # hidden size only applies to the line:
      #   attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
      # and won't affect Attention.__init__()

    # MLP part (QwenMLP)
    if hasattr(block, 'mlp'):
      block.mlp.gate_proj.tp_meta = TPMeta.COLUMN_NO_GATHER
      block.mlp.up_proj.tp_meta = TPMeta.COLUMN_NO_GATHER
      block.mlp.down_proj.tp_meta = TPMeta.ROW_NO_SCATTER
  return h


class SuikaQwenBase(nn.Module):
  """
  SuikaQwenBase model, a stack of QwenDecoderLayer modules,
  adapted from Hugging Face Qwen implementation, with support for tensor parallelism.
  """
  def __init__(self,
               config: QwenConfig,
               layer_index: int = 0,
               n_layer: int = -1, # if n_layer == -1, use config.num_hidden_layers
               adopt_Megatron_TP_flag: bool = True,
               init_on_meta: bool = False,
               ):
    super(SuikaQwenBase, self).__init__()
    self.config = config
    
    n_layer = n_layer if n_layer != -1 else config.num_hidden_layers

    with set_default_device_context("meta" if init_on_meta else None):
      self.h = nn.ModuleList([
          Qwen2DecoderLayer(config, layer_idx=layer_index + i) for i in range(n_layer)
      ])
      self.rotary_emb = Qwen2RotaryEmbedding(config=config)

    if layer_index != 0:
      self = replace_ModuleList_with_OffsetModuleList(self, layer_index)

    if adopt_Megatron_TP_flag:
      self.h = adopt_Megatron_TP_for_Qwen(self.h)

    self.position_embds_cache = None


  def forward(
        self,
        input_embds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        position_embds: Optional[torch.Tensor] = None,
    ):    
    hidden_states = input_embds

    if position_ids is not None:
      position_embds = self.rotary_emb(hidden_states, position_ids)
    elif position_embds is not None:
      pass
    else:
      position_embds = self._get_postition_embds(hidden_states)
    
    for block in self.h:
      layer_outputs = block(
          hidden_states,
          attention_mask=attention_mask,
          position_embeddings=position_embds,
      )
      hidden_states = layer_outputs[0]
      
    return hidden_states

  def _get_postition_embds(self, input_embds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if self.position_embds_cache is None:
      seq_len = input_embds.size(1)
      position_ids = torch.arange(0, seq_len, device=input_embds.device).unsqueeze(0)
      sin, cos = self.rotary_emb(input_embds, position_ids)
      sin.requires_grad_(False)
      cos.requires_grad_(False)
      self.position_embds_cache = (sin, cos)

    return self.position_embds_cache



class SuikaQwen(SuikaQwenBase):
  def __init__(
    self,
    config: QwenConfig,
    layer_index: int = 0,
    n_layer: int = -1, # if n_layer == -1, use config.num_hidden_layers
    add_embedding_and_lm_head: bool = True,
    init_on_meta: bool = False,
  ):
    super(SuikaQwen, self).__init__(
      config,
      layer_index=layer_index,
      n_layer=n_layer,
      adopt_Megatron_TP_flag=True,
      init_on_meta=init_on_meta,
    )
    
    with set_default_device_context("meta" if init_on_meta else None):
      n_layer = n_layer if n_layer != -1 else config.num_hidden_layers
      
      if layer_index == 0 and add_embedding_and_lm_head:
        # Qwen names wte as embed_tokens, but to keep it consistent with other models, we use wte
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
      else:
        self.wte = None
      
      if layer_index + n_layer == config.num_hidden_layers and add_embedding_and_lm_head:
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
      else:
        self.lm_head = None
        self.norm = None
        
      
  def forward(self, input_embds_or_ids, attention_mask=None, position_ids=None):
    if self.wte is not None:
      input_embds = self.wte(input_embds_or_ids)
    else:
      input_embds = input_embds_or_ids

    hidden_states = super(SuikaQwen, self).forward(
      input_embds, 
      attention_mask=attention_mask,
      position_ids=position_ids
    )
    
    if self.lm_head is not None:
      hidden_states = self.norm(hidden_states)
      lm_logits = self.lm_head(hidden_states)
      return lm_logits
    else:
      return hidden_states


def get_model_provider_Qwen(config: QwenConfig, **kwargs) -> Callable[[int, int], SuikaQwen]:
  def model_provider(layer_index, n_layer):
    model = SuikaQwen(config, layer_index, n_layer, **kwargs)
    return model
  return model_provider

# Example Qwen specifications
Qwen_seq_len = 2048

Qwen_specs: Dict[str, QwenConfig] = {
  "Qwen2-TEST": QwenConfig(
    hidden_size=1024,
    intermediate_size=2028, 
    num_hidden_layers=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=1024,
    max_position_embeddings=512,
    max_window_layers=8,
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
  ),
  "Qwen2-0.5B": QwenConfig(
    hidden_size=896,
    intermediate_size=4864,
    num_hidden_layers=24,
    num_attention_heads=14,
    num_key_value_heads=2,
    vocab_size=151936,
    max_position_embeddings=Qwen_seq_len,
    max_window_layers=24,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
  ),
  "Qwen2-1.5B": QwenConfig(
    hidden_size=1536,
    intermediate_size=8960,
    num_hidden_layers=28,
    num_attention_heads=12,
    num_key_value_heads=2,
    vocab_size=151936,
    max_position_embeddings=Qwen_seq_len,
    max_window_layers=28,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
  ),
  "Qwen2-7B": QwenConfig(
    hidden_size=3584,
    intermediate_size=18944,
    num_hidden_layers=28,
    num_attention_heads=28,
    num_key_value_heads=4,
    vocab_size=152064,
    max_position_embeddings=Qwen_seq_len,
    max_window_layers=28,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
  ),
  "Qwen2-72B": QwenConfig(
    hidden_size=8192,
    intermediate_size=29568,
    num_hidden_layers=80,
    num_attention_heads=64,
    num_key_value_heads=8,
    vocab_size=152064,
    max_position_embeddings=Qwen_seq_len,
    max_window_layers=80,
    rms_norm_eps=1e-5,
    tie_word_embeddings=False,
  ),
}

ModelConfig_Qwen_specs: Dict[str, ModelConfig] = {
  k: ModelConfig.from_huggingface(model_name=k, config=v)
  for k, v in Qwen_specs.items()
}


from Suika.cluster.resource import default_acc_type, testbed_half_nics, testbed_no_ib

ModelConfig_Qwen_specs["Qwen2-0.5B"] = \
  ModelConfig.from_huggingface(
    model_name="Qwen2-0.5B",
    config=Qwen_specs["Qwen2-0.5B"],
    estimator_C={
      'param': 28.443115234375, 'activation': 3.5, 'param_optim_mem': 113.7724609375, 'activation_mem_np': 17.5, 'activation_mem_p': 438.01534695095484, 'comp': 1.1580246390435833, 'optimizer': 0.7347692795169595, 'backward': 1.142818496916344, 'comp_commu_overlap': 1.0
    },
    rubick_perf_params={
      # 'k_bwd': 1.2724222210371392, 'k_sync': 1.023096692190767, 'k_opt': 0.11255197897227393, 'k_os_offload': 1.0, 'k_const': 0.004987181469447726, 'k_off': 1.0, 'k_swap': 1.0
      'k_bwd': 1.2724222210371392, 'k_sync': 1.023096692190767, 'k_opt': 0.11255197897227393, 'k_os_offload': 1.0, 'k_const': 0.004987181469447726, 'k_off': 1.0, 'k_swap': 1.0
    },
    sia_perf_params={
      'alpha_c': 0.04770168090663298, 'beta_c': 0.05131599528940201, 'alpha_n': 0.08789217386611589, 'beta_n': 0.007824499612438201, 'alpha_r': 0.07990197624192354, 'beta_r': 0.00711318146585291, 'gamma': 2.128398740169197
    } if not testbed_half_nics and not testbed_no_ib else ({
      'alpha_c': 0.04770168090663298, 'beta_c': 0.05131599528940201, 'alpha_n': 0.08789217386611589, 'beta_n': 0.007824499612438201, 'alpha_r': 0.07990197624192354, 'beta_r': 0.00711318146585291, 'gamma': 2.128398740169197
      } if not testbed_no_ib else {
      'alpha_c': 0.03389250107594571, 'beta_c': 0.05499500516406027, 'alpha_n': 1.5668334365110557, 'beta_n': 1e-08, 'alpha_r': 0.0921880249064013, 'beta_r': 0.0076478627216738536, 'gamma': 2.2880557688270695
      }
    ),
  )

ModelConfig_Qwen_specs["Qwen2-1.5B"] = \
  ModelConfig.from_huggingface(
    model_name="Qwen2-1.5B",
    config=Qwen_specs["Qwen2-1.5B"],
    estimator_C={
      'param': 89.259765625, 'activation': 6.0, 'param_optim_mem': 357.0390625, 'activation_mem_np': 30.0, 'activation_mem_p': 471.2401559012277, 'comp': 1.3076007256349726, 'optimizer': 1.27167782995819, 'backward': 1.3133803842019618, 'comp_commu_overlap': 1.0
    },
    rubick_perf_params={
      # 'k_bwd': 1.5345325349686136, 'k_sync': 10.0, 'k_opt': 0.011413923810495732, 'k_os_offload': 1.0, 'k_const': 100.0, 'k_off': 1.0, 'k_swap': 1.0
      'k_bwd': 1.6728915568505665, 'k_sync': 1.5067930918948578, 'k_opt': 0.04547954129675826, 'k_os_offload': 1.0, 'k_const': 0.028259731270472807, 'k_off': 1.0, 'k_swap': 1.0
    },
    sia_perf_params={
      'alpha_c': 0.05086192133722918, 'beta_c': 0.0720295377181298, 'alpha_n': 0.1102566585351768, 'beta_n': 0.0038149931054347625, 'alpha_r': 0.09888031821087152, 'beta_r': 0.00575308370606869, 'gamma': 1.4080117001136896
    } if not testbed_half_nics and not testbed_no_ib else ({
      'alpha_c': 0.05539590531638182, 'beta_c': 0.07033587616294582, 'alpha_n': 0.12049830076458899, 'beta_n': 3.268601840931073e-05, 'alpha_r': 0.09092881280960108, 'beta_r': 0.005515217695904498, 'gamma': 1.3389432866153899
      } if not testbed_no_ib else {
      'alpha_c': 0.05095577486134294, 'beta_c': 0.07074447500681717, 'alpha_n': 2.304599421490692, 'beta_n': 1e-08, 'alpha_r': 0.044060663970614054, 'beta_r': 0.00784910664605531, 'gamma': 1.0
      }
    ),
  )

ModelConfig_Qwen_specs["Qwen2-7B"] = \
  ModelConfig.from_huggingface(
    model_name="Qwen2-7B",
    config=Qwen_specs["Qwen2-7B"],
    estimator_C={
      'param': 444.5361328125, 'activation': 14.0, 'param_optim_mem': 1778.14453125, 'activation_mem_np': 70.0, 'activation_mem_p': 1113.4567173549108, 'comp': 3.9610776915961003, 'optimizer': 4.985017200521131, 'backward': 1.526670097260841, 'comp_commu_overlap': 1.0
    },
    rubick_perf_params={
      # 'k_bwd': 2.1644713451274193, 'k_sync': 1.0, 'k_opt': 0.015434595662421509, 'k_os_offload': 1.0, 'k_const': 0.008282993724328351, 'k_off': 1.0, 'k_swap': 1.0
      'k_bwd': 2.1644713451274193, 'k_sync': 1.0, 'k_opt': 0.015434595662421509, 'k_os_offload': 1.0, 'k_const': 0.008282993724328351, 'k_off': 1.0, 'k_swap': 1.0
    },
    sia_perf_params={
      'alpha_c': 0.10762796295968378, 'beta_c': 0.0948679153729161, 'alpha_n': 0.29085271077615193, 'beta_n': 0.00011150658562309025, 'alpha_r': 0.24538330781727657, 'beta_r': 1e-08, 'gamma': 1.55899502851992
    } if not testbed_half_nics and not testbed_no_ib else ({ 
      'alpha_c': 0.10723900889797061, 'beta_c': 0.09545842579693946, 'alpha_n': 0.19338031289436525, 'beta_n': 0.0002906182091285829, 'alpha_r': 0.14236658257039492, 'beta_r': 1.8148883342582942e-06, 'gamma': 1.0
      } if not testbed_no_ib else {
      'alpha_c': 0.10398735897202759, 'beta_c': 0.09659363032891156, 'alpha_n': 8.850830676180903, 'beta_n': 1e-08, 'alpha_r': 0.15067834050666626, 'beta_r': 1e-08, 'gamma': 1.0
      }
    ),
  )


ModelConfig_Qwen_specs["Qwen2-72B"] = \
  ModelConfig.from_huggingface(
    model_name="Qwen2-72B",
    config=Qwen_specs["Qwen2-72B"],
    estimator_C={
      'param': 1674.05078125, 'activation': 32.0, 'param_optim_mem': 6696.203125, 'activation_mem_np': 160.0, 'activation_mem_p': 2278.138427734375, 'comp': 9.135619447955063, 'optimizer': 15.816537385910436, 'backward': 1.524178546315581, 'comp_commu_overlap': 1.0
    },
    rubick_perf_params={
      'k_bwd': 2.047146672631224, 'k_sync': 1.0, 'k_opt': 0.0463677682458259, 'k_os_offload': 1.0, 'k_const': 0.0013273362208321851, 'k_off': 1.0, 'k_swap': 1.0
    },
    sia_perf_params={
      'alpha_c': 0.34619999718864714, 'beta_c': 1e-08, 'alpha_n': 0.3338634698048955, 'beta_n': 0.010627078944651432, 'alpha_r': 1e-08, 'beta_r': 1e-08, 'gamma': 1.0441514550244277
    } 
  )


ModelStrategy_Qwen_specs: Dict[str, ModelStrategy] = {
  "Qwen2-0.5B": ModelStrategy(
      model_config=ModelConfig_Qwen_specs["Qwen2-0.5B"],
      acc_type=default_acc_type,
      global_batch_size=12,
      num_micro_batch=2,
      pp_dp_tp=(1, 1, 1),
      iter_time=0.727
  ),
  "Qwen2-1.5B": ModelStrategy(
      model_config=ModelConfig_Qwen_specs["Qwen2-1.5B"],
      acc_type=default_acc_type,
      global_batch_size=64,
      num_micro_batch=4,
      pp_dp_tp=(1, 4, 1),
      iter_time=1.480
  ),
  "Qwen2-7B": ModelStrategy(
      model_config=ModelConfig_Qwen_specs["Qwen2-7B"],
      acc_type=default_acc_type,
      global_batch_size=16,
      num_micro_batch=2,
      pp_dp_tp=(1, 4, 2),
      iter_time=0.729
  ),
  "Qwen2-72B": ModelStrategy(
      model_config=ModelConfig_Qwen_specs["Qwen2-72B"],
      acc_type=default_acc_type,
      global_batch_size=128,
      num_micro_batch=32,
      pp_dp_tp=(2, 4, 8),
      iter_time=6.0436
  ),
}