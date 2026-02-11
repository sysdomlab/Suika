from Suika.models.utils import (
  ModelConfig, ModelStrategy,
  replace_ModuleList_with_OffsetModuleList,
)

from Suika.tensor_parallel import TPMeta
from Suika.utils import set_default_device_context

from typing import Callable, Dict, Optional, Tuple
import torch
import torch.nn as nn

from transformers.models.llama.modeling_llama import LlamaConfig, LlamaDecoderLayer, LlamaRotaryEmbedding, LlamaRMSNorm


def adopt_Megatron_TP_for_Llama(h: nn.ModuleList):
  """
  Adapts LlamaDecoderLayers for Megatron-style Tensor Parallelism.
  Sets tp_meta attributes on relevant nn.Linear modules within LlamaAttention and LlamaMLP.
  """
  for block in h:
    # Self Attention 
    if hasattr(block, 'self_attn'):
      block.self_attn.q_proj.tp_meta = TPMeta.COLUMN_NO_GATHER
      block.self_attn.k_proj.tp_meta = TPMeta.COLUMN_NO_GATHER
      block.self_attn.v_proj.tp_meta = TPMeta.COLUMN_NO_GATHER
      block.self_attn.o_proj.tp_meta = TPMeta.ROW_NO_SCATTER
      block.self_attn.div_by_tp_deg = ["num_heads", "num_key_value_heads"]

    # MLP part (LlamaMLP)
    if hasattr(block, 'mlp'):
      block.mlp.gate_proj.tp_meta = TPMeta.COLUMN_NO_GATHER
      block.mlp.up_proj.tp_meta = TPMeta.COLUMN_NO_GATHER
      block.mlp.down_proj.tp_meta = TPMeta.ROW_NO_SCATTER
  return h


class SuikaLlamaBase(nn.Module):
  """
  SuikaLlama model, a stack of LlamaDecoderLayer modules,
  adapted from Hugging Face Llama implementation, with support for tensor parallelism.
  """
  def __init__(self,
               config: LlamaConfig,
               layer_index: int = 0,
               n_layer: int = -1, # if n_layer == -1, use config.n_layer
               adopt_Megatron_TP_flag: bool = True,
               init_on_meta: bool = False,
               ):
    super(SuikaLlamaBase, self).__init__()
    self.config = config
    
    n_layer = n_layer if n_layer != -1 else config.num_hidden_layers

    with set_default_device_context("meta" if init_on_meta else None):
      self.h = nn.ModuleList([
          LlamaDecoderLayer(config, layer_idx=layer_index + i) for i in range(n_layer)
      ])
      self.rotary_emb = LlamaRotaryEmbedding(config=config)

    if layer_index != 0:
      self = replace_ModuleList_with_OffsetModuleList(self, layer_index)

    if adopt_Megatron_TP_flag:
      self.h = adopt_Megatron_TP_for_Llama(self.h)

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
    
    assert not position_embds[0].requires_grad, "position_embds should not require gradients"

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

class SuikaLlama(SuikaLlamaBase):
  def __init__(
    self,
    config: LlamaConfig,
    layer_index: int = 0,
    n_layer: int = -1, # if n_layer == -1, use config.n_layer
    add_embedding_and_lm_head: bool = True,
    init_on_meta: bool = False,
  ):
    super(SuikaLlama, self).__init__(
      config,
      layer_index=layer_index,
      n_layer=n_layer,
      adopt_Megatron_TP_flag=True,
      init_on_meta=init_on_meta,
    )
    
    with set_default_device_context("meta" if init_on_meta else None):
      n_layer = n_layer if n_layer != -1 else config.num_hidden_layers
      
      if layer_index == 0 and add_embedding_and_lm_head:
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
      else:
        self.wte = None
      
      if layer_index + n_layer == config.num_hidden_layers and add_embedding_and_lm_head:
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.norm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
      else:
        self.lm_head = None
        self.norm = None
      
  def forward(self, input_embds_or_ids, attention_mask=None, position_ids=None):
    if self.wte is not None:
      input_embds = self.wte(input_embds_or_ids)
    else:
      input_embds = input_embds_or_ids
      
    hidden_states = super(SuikaLlama, self).forward(
      input_embds, 
      attention_mask=attention_mask,
      position_ids=position_ids,
    )
    
    if self.lm_head is not None:
      hidden_states = self.norm(hidden_states)
      lm_logits = self.lm_head(hidden_states)
      return lm_logits
    else:
      return hidden_states


def get_model_provider_Llama(config: LlamaConfig, **kwargs) -> Callable[[int, int], SuikaLlama]:
  def model_provider(layer_index, n_layer):
    model = SuikaLlama(config, layer_index, n_layer, **kwargs)
    return model
  return model_provider

Llama_seq_len = 2048

Llama_specs: Dict[str, LlamaConfig] = {
  "Llama-TEST": LlamaConfig(
    hidden_size=16,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    vocab_size=128,
    max_position_embeddings=16,
  ),
  "Llama-100M": LlamaConfig(
    hidden_size=768,
    intermediate_size=2048,
    num_hidden_layers=12,
    num_attention_heads=12,
    vocab_size=32000,
    max_position_embeddings=Llama_seq_len,
  ),
  "Llama-300M": LlamaConfig(
      hidden_size=1024,
      intermediate_size=2730,
      num_hidden_layers=16,
      num_attention_heads=16,
      vocab_size=32000,
      max_position_embeddings=Llama_seq_len,
  ),
  "Llama-7B": LlamaConfig(
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32,
    vocab_size=32000,
    max_position_embeddings=Llama_seq_len,
  ),
  "Llama-13B": LlamaConfig(
    hidden_size=5120,
    intermediate_size=13824,
    num_hidden_layers=40,
    num_attention_heads=40,
    vocab_size=32000,
    max_position_embeddings=Llama_seq_len,
  ),
  "Llama-30B": LlamaConfig(
    hidden_size=6656,
    intermediate_size=17920,
    num_hidden_layers=60,
    num_attention_heads=52,
    vocab_size=32000,
    max_position_embeddings=Llama_seq_len,
  ),
  "Llama-65B": LlamaConfig(
    hidden_size=8192,
    intermediate_size=22016,
    num_hidden_layers=80,
    num_attention_heads=64,
    vocab_size=32000,
    max_position_embeddings=Llama_seq_len,
  ),
  "Llama2-7B": LlamaConfig(
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=32,
    vocab_size=32000,
    max_position_embeddings=Llama_seq_len,
    hidden_act="silu"
  ),
  "Llama2-13B": LlamaConfig(
    hidden_size=5120,
    intermediate_size=13824,
    num_hidden_layers=40,
    num_attention_heads=40,
    num_key_value_heads=40,
    vocab_size=32000,
    max_position_embeddings=Llama_seq_len,
    hidden_act="silu"
  ),
  "Llama2-70B": LlamaConfig(
    hidden_size=8192,
    intermediate_size=28672,
    num_hidden_layers=80,
    num_attention_heads=64,
    num_key_value_heads=8,
    vocab_size=32000,
    max_position_embeddings=Llama_seq_len,
    hidden_act="silu"
  )
}


ModelConfig_Llama_specs: Dict[str, ModelConfig] = {
  k: ModelConfig.from_huggingface(model_name=k, config=v) for k, v in Llama_specs.items()
}

from Suika.cluster.resource import default_acc_type, testbed_half_nics, testbed_no_ib


ModelConfig_Llama_specs["Llama2-7B"] = \
  ModelConfig.from_huggingface(
    model_name="Llama2-7B",
    config=Llama_specs["Llama2-7B"],
    estimator_C={
      'param': 386.03125, 'activation': 16.0, 'param_optim_mem': 1544.125, 'activation_mem_np': 128.0, 'activation_mem_p': 1013.7856750488281, 'comp': 4.2596357946909835, 'optimizer': 4.179762514190285, 'backward': 1.5071714168714145, 'comp_commu_overlap': 1.0
    },
    rubick_perf_params={
      # 'k_bwd': 2.716572967337959, 'k_sync': 1.0, 'k_opt': 1e-08, 'k_os_offload': 1.0, 'k_const': 0.006748263707781823, 'k_off': 1.0, 'k_swap': 1.0
      'k_bwd': 2.716572967337959, 'k_sync': 1.0, 'k_opt': 1e-08, 'k_os_offload': 1.0, 'k_const': 0.006748263707781823, 'k_off': 1.0, 'k_swap': 1.0
    },
    sia_perf_params={
      'alpha_c': 0.01421001162274573, 'beta_c': 0.17602542075247174, 'alpha_n': 0.193335085078603, 'beta_n': 0.019142543779950446, 'alpha_r': 0.206179368424576, 'beta_r': 0.0028577648327236674, 'gamma': 1.451027003674824
    } if not testbed_half_nics and not testbed_no_ib else ({
      'alpha_c': 0.018053716765178657, 'beta_c': 0.17086284248777453, 'alpha_n': 0.24768250838076306, 'beta_n': 5.7743956482582576e-05, 'alpha_r': 0.15937124224591062, 'beta_r': 3.244099484969765e-05, 'gamma': 1.1370108452222791
      } if not testbed_no_ib else {
      'alpha_c': 0.013141860094616282, 'beta_c': 0.1748545738682775, 'alpha_n': 1.8050261266756247, 'beta_n': 0.7569956231819133, 'alpha_r': 0.2022694354792552, 'beta_r': 0.0031891936776733, 'gamma': 1.4105636515727924
      }
    ),
  )

ModelConfig_Llama_specs["Llama2-13B"] = \
  ModelConfig.from_huggingface(
    model_name="Llama2-13B",
    config=Llama_specs["Llama2-13B"],
    estimator_C={
      'param': 605.078125, 'activation': 20.0, 'param_optim_mem': 2420.3125, 'activation_mem_np': 160.0, 'activation_mem_p': 1268.559033203125, 'comp': 6.558003237756403, 'optimizer': 6.892964988126173, 'backward': 1.6698730538279216, 'comp_commu_overlap': 1.6646728515625
    },
    rubick_perf_params={
      # 'k_bwd': 2.142234127854154, 'k_sync': 1.0, 'k_opt': 0.030299127453248805, 'k_os_offload': 1.0, 'k_const': 0.0058849410859459685, 'k_off': 1.0, 'k_swap': 1.0
      'k_bwd': 2.142234127854154, 'k_sync': 1.0, 'k_opt': 0.030299127453248805, 'k_os_offload': 1.0, 'k_const': 0.0058849410859459685, 'k_off': 1.0, 'k_swap': 1.0
    },
    sia_perf_params={
      'alpha_c': 0.04777740539899172, 'beta_c': 0.17075317652278746, 'alpha_n': 0.3003040406618119, 'beta_n': 1e-08, 'alpha_r': 1e-08, 'beta_r': 1e-08, 'gamma': 1.0973510146698704
      # 'alpha_c': 0.05109999896782844, 'beta_c': 0.17050000082772032, 'alpha_n': 0.1509326986307661, 'beta_n': 0.08244628781369125, 'alpha_r': 1e-08, 'beta_r': 1e-08, 'gamma': 1.0199429889195009
    } if not testbed_half_nics and not testbed_no_ib else ({
      'alpha_c': 0.05018284060945162, 'beta_c': 0.1675413886483735, 'alpha_n': 0.17914285309528707, 'beta_n': 0.07763619653647773, 'alpha_r': 1e-08, 'beta_r': 1e-08, 'gamma': 1.1118655942681643
      } if not testbed_no_ib else {
      'alpha_c': 0.08593153761243796, 'beta_c': 0.12596570445329222, 'alpha_n': 7.485662429237848, 'beta_n': 1e-08, 'alpha_r': 3.4888215552495786e-06, 'beta_r': 1e-08, 'gamma': 1.0
      }
    ),
  )

ModelConfig_Llama_specs["Llama-30B"] = \
  ModelConfig.from_huggingface(
    model_name="Llama-30B",
    config=Llama_specs["Llama-30B"],
    estimator_C={
      'param': 1020.6015625, 'activation': 26.0, 'param_optim_mem': 4082.40625, 'activation_mem_np': 208.0, 'activation_mem_p': 1670.8811197916666, 'comp': 9.301742708705564, 'optimizer': 10.302032220865886, 'backward': 1.7192397799755248, 'comp_commu_overlap': 1.65093994140625
    },
    rubick_perf_params={
      'k_bwd': 1.0, 'k_sync': 1.1631768064219403, 'k_opt': 0.07967280386750876, 'k_os_offload': 1.0, 'k_const': 1e-08, 'k_off': 1.0, 'k_swap': 1.0
    },
    sia_perf_params={
      'alpha_c': 0.17220833333333332, 'beta_c': 0.1736962065039171, 'alpha_n': 0.24444270819847339, 'beta_n': 1e-08, 'alpha_r': 0.07918108546567584, 'beta_r': 1e-08, 'gamma': 1.0093390510081073
    }
  )

ModelConfig_Llama_specs["Llama-65B"] = \
  ModelConfig.from_huggingface(
    model_name="Llama-65B",
    config=Llama_specs["Llama-65B"],
    estimator_C={
      'param': 1544.03125, 'activation': 32.0, 'param_optim_mem': 6176.125, 'activation_mem_np': 256.0, 'activation_mem_p': 2064.1380615234375, 'comp': 9.042879487028696, 'optimizer': 14.5379178148872, 'backward': 1.4973483774152196, 'comp_commu_overlap': 1.0
    },
    rubick_perf_params={
      'k_bwd': 2.4164501063507227, 'k_sync': 10.0, 'k_opt': 1e-08, 'k_os_offload': 1.0, 'k_const': 0.957387791112509, 'k_off': 1.0, 'k_swap': 1.0
    },
    sia_perf_params={
      'alpha_c': 0.3413999973370423, 'beta_c': 1e-08, 'alpha_n': 0.28969499811647914, 'beta_n': 0.009557467925680085, 'alpha_r': 1e-08, 'beta_r': 1e-08, 'gamma': 1.0005826152801895
    }
  )

ModelConfig_Llama_specs["Llama2-70B"] = \
  ModelConfig.from_huggingface(
    model_name="Llama2-70B",
    config=Llama_specs["Llama2-70B"],
    estimator_C={
      'param': 1632.03125, 'activation': 32.0, 'param_optim_mem': 6528.125, 'activation_mem_np': 256.0, 'activation_mem_p': 2168.1380615234375, 'comp': 9.025805290115741, 'optimizer': 15.352805946229632, 'backward': 1.5156583243696422, 'comp_commu_overlap': 1.0
    },
    rubick_perf_params={
      'k_bwd': 2.438079378099727, 'k_sync': 10.0, 'k_opt': 1e-08, 'k_os_offload': 1.0, 'k_const': 100.0, 'k_off': 1.0, 'k_swap': 1.0
    },
    sia_perf_params={
      'alpha_c': 0.3425999987260668, 'beta_c': 1e-08, 'alpha_n': 0.23668528136419262, 'beta_n': 0.0016759683952951093, 'alpha_r': 1e-08, 'beta_r': 1e-08, 'gamma': 1.0000019500576884
    }
  )


ModelStrategy_Llama_specs: Dict[str, ModelStrategy] = {
  "Llama2-7B": ModelStrategy(
    model_config=ModelConfig_Llama_specs["Llama2-7B"],
    acc_type=default_acc_type,
    global_batch_size=32,
    num_micro_batch=4,
    pp_dp_tp=(1, 4, 2),
    iter_time=1.581
  ),
  "Llama2-13B": ModelStrategy(
    model_config=ModelConfig_Llama_specs["Llama2-13B"],
    acc_type=default_acc_type,
    global_batch_size=32,
    num_micro_batch=4,
    pp_dp_tp=(1, 4, 4),
    iter_time=1.912
  ) if not testbed_no_ib else ModelStrategy(
    model_config=ModelConfig_Llama_specs["Llama2-13B"],
    acc_type=default_acc_type,
    global_batch_size=30,
    num_micro_batch=5,
    pp_dp_tp=(2, 2, 4),
    iter_time=1.916
  ),
  "Llama-30B": ModelStrategy(
    model_config=ModelConfig_Llama_specs["Llama-30B"],
    acc_type=default_acc_type,
    global_batch_size=32,
    num_micro_batch=8,
    pp_dp_tp=(2, 4, 4),
    iter_time=2.280
  ),
  "Llama-65B": ModelStrategy(
    model_config=ModelConfig_Llama_specs["Llama-65B"],
    acc_type=default_acc_type,
    global_batch_size=128,
    num_micro_batch=32,
    pp_dp_tp=(2, 4, 8),
    iter_time=5.9381
  ),
  "Llama2-70B": ModelStrategy(
    model_config=ModelConfig_Llama_specs["Llama2-70B"],
    acc_type=default_acc_type,
    global_batch_size=128,
    num_micro_batch=16,
    pp_dp_tp=(3, 8, 8),
    iter_time=2.3224
  ),
}