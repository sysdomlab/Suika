from Suika.models.utils import ModelStrategy, replace_ModuleList_with_OffsetModuleList, replace_Conv1D, ModelConfig
from Suika.tensor_parallel import TPMeta

from Suika.utils import set_default_device_context

from typing import Callable, Dict
import torch.nn as nn

from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Config

GPTConfig = GPT2Config

def adopt_Megatron_TP_for_GPT(h: nn.ModuleList):
  """
  See Megatron https://arxiv.org/pdf/1909.08053
  """
  for block in h:
    """
    self attention
    """
    block.attn.c_attn.tp_meta = TPMeta.COLUMN_NO_GATHER
    block.attn.c_proj.tp_meta = TPMeta.ROW_NO_SCATTER
    """
    MLP
    """
    block.mlp.c_fc.tp_meta = TPMeta.COLUMN_NO_GATHER
    block.mlp.c_proj.tp_meta = TPMeta.ROW_NO_SCATTER
    # hugging face implementation uses num_heads & split_size for multi head attention.
    block.attn.div_by_tp_deg = ["num_heads", "split_size"]

  return h


"""
GPT2 model of only transformer blocks.

Adapted from hugging face GPT2 implementation, support tensor parallelism.

To use Suika's TP impl, repalce_HuggingfaceConv1D_flag must be True
To use Suika's migration impl, layer_index of the first block must be set

To enable efficient TP, adopt_Megatron_TP_flag should be set
"""
class SuikaGPTBase(nn.Module):
  def __init__(self, 
               config: GPTConfig, 
               layer_index = 0,
               n_layer = -1, # if n_layer == -1, use config.n_layer
               replace_HuggingfaceConv1D_flag = True, 
               adopt_Megatron_TP_flag = True,
               init_on_meta = False,
               copy_param_from_HuggingfaceConv1D = False,
               ):
    super(SuikaGPTBase, self).__init__()
    self.config = config
    
    n_layer = n_layer if n_layer != -1 else config.n_layer

    with set_default_device_context("meta" if init_on_meta else None):
      h = nn.ModuleList([
        GPT2Block(config) for _ in range(n_layer)
      ])
    
    self.h = h
    
    if layer_index != 0:
      self = replace_ModuleList_with_OffsetModuleList(self, layer_index)

    if replace_HuggingfaceConv1D_flag:
      self.h = replace_Conv1D(self.h, init_on_meta, copy_param_from_HuggingfaceConv1D)

    if adopt_Megatron_TP_flag:
      self.h = adopt_Megatron_TP_for_GPT(self.h)

  def forward(self, input_embds, attention_mask=None):
    # input_embds: (batch_size, seq, hidden)
    hidden_states = input_embds

    for block in self.h:
      hidden_states = block(hidden_states, attention_mask=attention_mask)[0]
      
    return hidden_states
  
"""
Complete GPT2 model with embedding and lm_head.
"""
class SuikaGPT(SuikaGPTBase):
  
  def __init__(
    self,
    config: GPTConfig,
    layer_index: int = 0,
    n_layer: int = -1, # if n_layer == -1, use config.n_layer
    add_embedding_and_lm_head: bool = True,
    init_on_meta: bool = False,
  ):
    super(SuikaGPT, self).__init__(
      config,
      layer_index=layer_index,
      n_layer=n_layer,
      replace_HuggingfaceConv1D_flag=True,
      adopt_Megatron_TP_flag=True,
      init_on_meta=init_on_meta,
      copy_param_from_HuggingfaceConv1D=not init_on_meta,
    )
  
    with set_default_device_context("meta" if init_on_meta else None):
      n_layer = n_layer if n_layer != -1 else config.n_layer
      
      if layer_index == 0 and add_embedding_and_lm_head:
        # add embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
      else:
        self.wte = None
        
      if layer_index + n_layer == config.n_layer and add_embedding_and_lm_head:
        # add lm_head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
      else:
        self.lm_head = None
    
  def forward(self, input_embds_or_ids, attention_mask=None):
    # input_embds_or_ids: (batch_size, seq, hidden) or (batch_size, seq)
    if self.wte is None:
      input_embds = input_embds_or_ids
    else:
      input_embds = self.wte(input_embds_or_ids)
    
    hidden_states = super(SuikaGPT, self).forward(input_embds, attention_mask=attention_mask)
    
    if self.lm_head is None:
      return hidden_states
    else:
      lm_logits = self.lm_head(hidden_states)
      # output is ( batch_size, seq, vocab_size ), to pass to CrossEntropyLoss, flatten is needed
      return lm_logits


"""
Get model provider function for GPT
"""
def get_model_provider_GPT(config: GPTConfig, **kwargs) -> Callable[[int, int], SuikaGPT]:
  def model_provider(layer_index, n_layer):
    model = SuikaGPT(config, layer_index, n_layer, **kwargs)
    return model
  return model_provider

"""
Predefined GPT2 model specs of GPTConfigs
"""
GPT_specs: Dict[str, GPTConfig] = {
  "GPT-TEST": GPTConfig(
    vocab_size=128,
    n_positions = 16,
    n_embd = 16,
    n_head = 4,
    n_layer = 4,
  ),
  "GPT-125M": GPTConfig(
    vocab_size=51200,
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
  ),
  "GPT-350M": GPTConfig(
    vocab_size=51200,
    n_positions=1024,
    n_embd=1024,
    n_layer=24,
    n_head=16,
  ),
  "GPT-510M": GPTConfig(
    vocab_size=51200,
    n_positions=1024,
    n_embd=1248,
    n_layer=24,
    n_head=16,
  ),
  "GPT-760M": GPTConfig(
    vocab_size=51200,
    n_positions=1024,
    n_embd=1536,
    n_layer=24,
    n_head=16,
  ),
  "GPT-1.3B": GPTConfig(
    vocab_size=51200,
    n_positions=1024,
    n_embd=2048,
    n_layer=24,
    n_head=32,
  ),
  "GPT-2.6B": GPTConfig(
    vocab_size=51200,
    n_positions=1024,
    n_embd=2560,
    n_layer=32,
    n_head=32,
  ),
  "GPT-6.7B": GPTConfig(
    vocab_size=51200,
    n_positions=1024,
    n_embd=4096,
    n_layer=32,
    n_head=32,
  ),
  "GPT-15B": GPTConfig(
    vocab_size=51200,
    n_positions=1024,
    n_embd=5120,
    n_layer=48,
    n_head=40,
  ),
  "GPT-39B": GPTConfig(
    vocab_size=51200,
    n_positions=1024,
    n_embd=8192,
    n_layer=48,
    n_head=64,
  ),
  "GPT-76B": GPTConfig(
    vocab_size=51200,
    n_positions=1024,
    n_embd=10240,
    n_layer=60,
    n_head=80,
  ),
  "GPT-175B": GPTConfig(
    vocab_size=51200,
    n_positions=1024,
    n_embd=12288,
    n_layer=96,
    n_head=96,
  ),
}

"""
Predefined GPT2 model specs of ModelConfig
"""
ModelConfig_GPT_specs: Dict[str, ModelConfig] = {
  k: ModelConfig.from_huggingface(model_name=k, config=v) for k, v in GPT_specs.items()
}

"""
Some of the specs are used in final experiments, and their detailed performance data are listed below:

Only specs in final experiments are available in ModelStrategy_GPT_specs

Pre-profiled estimator_C here is only for:
1. estimate plan feasibility for Rubick / Sia
2. simulation experiments
Suika always acquires estimator constants at runtime without prior profiling
"""

from Suika.cluster.resource import default_acc_type, testbed_half_nics, testbed_no_ib

ModelConfig_GPT_specs["GPT-350M"] = \
  ModelConfig.from_huggingface(
    model_name="GPT-350M",
    config=GPT_specs["GPT-350M"],
    estimator_C={
      'param': 24.025390625, 'activation': 2.0, 'param_optim_mem': 96.1015625, 'activation_mem_np': 10.0, 'activation_mem_p': 128.49522018432617, 'comp': 0.30617576204132235, 'optimizer': 0.5725161207697126, 'backward': 1.4442248882275956, 'comp_commu_overlap': 1.5
    },
    rubick_perf_params={
      # 'k_bwd': 1.1612502928466804, 'k_sync': 10.0, 'k_opt': 1e-08, 'k_os_offload': 1.0, 'k_const': 76.00161429854226, 'k_off': 1.0, 'k_swap': 1.0
      'k_bwd': 1.1612502928466804, 'k_sync': 10.0, 'k_opt': 1e-08, 'k_os_offload': 1.0, 'k_const': 76.00161429854226, 'k_off': 1.0, 'k_swap': 1.0
    },
    sia_perf_params={ # cluster-a
      'alpha_c': 0.010974260176270366, 'beta_c': 0.017979710062468224, 'alpha_n': 0.06269152349284741, 'beta_n': 1.1000000000000001e-08, 'alpha_r': 0.05699229408440674, 'beta_r': 1e-08, 'gamma': 1.5360820346603787
    } if not testbed_half_nics and not testbed_no_ib else ({ # cluster-b half nics
        'alpha_c': 0.010974260176270366, 'beta_c': 0.017979710062468224, 'alpha_n': 0.06269152349284741, 'beta_n': 1.1000000000000001e-08, 'alpha_r': 0.05699229408440674, 'beta_r': 1e-08, 'gamma': 1.5360820346603787
      } if not testbed_no_ib else { # cluster-c no ib
        'alpha_c': 0.02188084670657014, 'beta_c': 0.016680726853755753, 'alpha_n': 0.8565581826768423, 'beta_n': 0.002932403465364653, 'alpha_r': 0.021215513658017403, 'beta_r': 0.0034625779212967287, 'gamma': 1.000445395004087
      }
    ),
  )
ModelConfig_GPT_specs["GPT-1.3B"] = \
  ModelConfig.from_huggingface(
    model_name="GPT-1.3B",
    config=GPT_specs["GPT-1.3B"],
    estimator_C={
      'param': 96.05078125, 'activation': 4.0, 'param_optim_mem': 384.203125, 'activation_mem_np': 20.0, 'activation_mem_p': 256.97481536865234, 'comp': 0.6672308342132016, 'optimizer': 1.2530193379878993, 'backward': 1.4734644160496673, 'comp_commu_overlap': 1
    },
    rubick_perf_params={
      # 'k_bwd': 1.3156442341568273, 'k_sync': 1.2237688662061021, 'k_opt': 1e-08, 'k_os_offload': 1.0, 'k_const': 90.57797967886678, 'k_off': 1.0, 'k_swap': 1.0
      'k_bwd': 1.337660570762399, 'k_sync': 9.29903586167245, 'k_opt': 0.0014664287073894637, 'k_os_offload': 1.0, 'k_const': 89.51498625738627, 'k_off': 1.0, 'k_swap': 1.0
    },
    sia_perf_params={
      'alpha_c': 0.019694179394189687, 'beta_c': 0.036392831684522196, 'alpha_n': 0.07553073330251578, 'beta_n': 0.001111253050776635, 'alpha_r': 0.0620536032312496, 'beta_r': 0.002232357471512203, 'gamma': 1.1252910370382725
    } if not testbed_half_nics and not testbed_no_ib else ({
      'alpha_c': 0.021283622533608836, 'beta_c': 0.036035449554481895, 'alpha_n': 0.07259387546781972, 'beta_n': 0.0017951405226617624, 'alpha_r': 0.06051692805193818, 'beta_r': 0.0021547676703127265, 'gamma': 1.1129996849077837
      } if not testbed_no_ib else {
      'alpha_c': 0.016551128385440872, 'beta_c': 0.037343956670531905, 'alpha_n': 0.8132891754077334, 'beta_n': 0.05441106216661124, 'alpha_r': 0.056792722297513365, 'beta_r': 0.002720741061974834, 'gamma': 1.0812355092835775
      }
    ),
  )
ModelConfig_GPT_specs["GPT-2.6B"] = \
  ModelConfig.from_huggingface(
    model_name="GPT-2.6B",
    config=GPT_specs["GPT-2.6B"],
    estimator_C={
      'param': 150.0634765625, 'activation': 5.0, 'param_optim_mem': 600.25390625, 'activation_mem_np': 25.0, 'activation_mem_p': 281.36890029907227, 'comp': 0.7908851400648607, 'optimizer': 1.71733494027257, 'backward': 1.5357509216019625, 'comp_commu_overlap': 1
    },
    rubick_perf_params={
      # 'k_bwd': 1.3474908648119261, 'k_sync': 9.404181877729085, 'k_opt': 0.011749606876879178, 'k_os_offload': 1.0, 'k_const': 100.0, 'k_off': 1.0, 'k_swap': 1.0
      'k_bwd': 1.5023717776872436, 'k_sync': 10.0, 'k_opt': 0.005652403655203042, 'k_os_offload': 1.0, 'k_const': 100.0, 'k_off': 1.0, 'k_swap': 1.0
    },
    sia_perf_params={
      'alpha_c': 0.026370524331109327, 'beta_c': 0.05727753942939865, 'alpha_n': 0.09551742451006613, 'beta_n': 0.003874546138122705, 'alpha_r': 0.11685242287217823, 'beta_r': 0.0016136402513682053, 'gamma': 1.1090322715065664
    } if not testbed_half_nics and not testbed_no_ib else ({
      'alpha_c': 0.03161373689292622, 'beta_c': 0.05542469753343327, 'alpha_n': 0.09949703988487375, 'beta_n': 0.0036562492367768545, 'alpha_r': 0.10612892266559208, 'beta_r': 0.001569039587552782, 'gamma': 1.0465085265836185
      } if not testbed_no_ib else {
      'alpha_c': 0.019068701474981544, 'beta_c': 0.060307391451277934, 'alpha_n': 2.6673831255582074, 'beta_n': 0.046471403552423865, 'alpha_r': 0.13104197490646224, 'beta_r': 0.001643776830253534, 'gamma': 1.2064456769719336
      }
    ),
  )

ModelConfig_GPT_specs["GPT-6.7B"] = \
  ModelConfig.from_huggingface(
    model_name="GPT-6.7B",
    config=GPT_specs["GPT-6.7B"],
    estimator_C={
      'param': 384.1484375, 'activation': 8.0, 'param_optim_mem': 1536.59375, 'activation_mem_np': 40.0, 'activation_mem_p': 355.03467559814453, 'comp': 1.2951777683729895, 'optimizer': 4.190875726540882, 'backward': 1.6288200814441816, 'comp_commu_overlap': 1
    },
    rubick_perf_params={
      # 'k_bwd': 2.7780165714020644, 'k_sync': 1.0, 'k_opt': 1e-08, 'k_os_offload': 1.0, 'k_const': 0.020505321126278776, 'k_off': 1.0, 'k_swap': 1.0
      'k_bwd': 2.7780165714020644, 'k_sync': 1.0, 'k_opt': 1e-08, 'k_os_offload': 1.0, 'k_const': 0.020505321126278776, 'k_off': 1.0, 'k_swap': 1.0
    },
    sia_perf_params={
      'alpha_c': 0.04471388794508958, 'beta_c': 0.044429219830759537, 'alpha_n': 0.22583711631907274, 'beta_n': 1e-08, 'alpha_r': 0.12333073496569555, 'beta_r': 0.0006154373879139035, 'gamma': 1.0069671262074154
    } if not testbed_half_nics and not testbed_no_ib else ({
      'alpha_c': 0.04863697237131869, 'beta_c': 0.0436453079808178, 'alpha_n': 0.12707660276613375, 'beta_n': 0.016657125668106308, 'alpha_r': 0.11803237085052252, 'beta_r': 0.001794777867312126, 'gamma': 1.0
      } if not testbed_no_ib else {
      'alpha_c': 0.028711541635538906, 'beta_c': 0.050366889103239054, 'alpha_n': 7.400002233460978, 'beta_n': 0.0016862491266922775, 'alpha_r': 0.12385358259826192, 'beta_r': 2.451900029861262e-05, 'gamma': 1.008887502706942
      }
    ),
  )

ModelConfig_GPT_specs["GPT-15B"] = \
  ModelConfig.from_huggingface(
    model_name="GPT-15B",
    config=GPT_specs["GPT-15B"],
    estimator_C={      
      'param': 600.302734375, 'activation': 10.0, 'param_optim_mem': 2401.2109375, 'activation_mem_np': 50.0, 'activation_mem_p': 447.50218370225696, 'comp': 1.9063465938199635, 'optimizer': 7.246891870159573, 'backward': 1.8402750042772817, 'comp_commu_overlap': 1.4010009765625
    },
    rubick_perf_params={
      # 'k_bwd': 1.7928448395094987, 'k_sync': 1.0, 'k_opt': 0.03661878384550206, 'k_os_offload': 1.0, 'k_const': 0.00417352786432772, 'k_off': 1.0, 'k_swap': 1.0
      'k_bwd': 1.6972705598184865, 'k_sync': 1.0, 'k_opt': 0.03678337574071851, 'k_os_offload': 1.0, 'k_const': 0.005421368263422922, 'k_off': 1.0, 'k_swap': 1.0
    },
    sia_perf_params={
      'alpha_c': 0.09006013734775713, 'beta_c': 0.04963127347412736, 'alpha_n': 0.4081749200063147, 'beta_n': 1e-08, 'alpha_r': 1e-08, 'beta_r': 1e-08, 'gamma': 1.0698893719741203
    } if not testbed_half_nics and not testbed_no_ib else ({
      'alpha_c': 0.11558808775933034, 'beta_c': 0.03781583751896219, 'alpha_n': 0.2215542536233474, 'beta_n': 0.09079575483995589, 'alpha_r': 1e-08, 'beta_r': 1e-08, 'gamma': 1.0739338463636519
      }if not testbed_no_ib else {
      'alpha_c': 0.1369823545491944, 'beta_c': 0.05233012823338547, 'alpha_n': 15.763666132033366, 'beta_n': 1e-08, 'alpha_r': 0.06695949940391556, 'beta_r': 7.281365168129002e-07, 'gamma': 1.0
      }
    ),
  )

ModelConfig_GPT_specs["GPT-39B"] = \
  ModelConfig.from_huggingface(
    model_name="GPT-39B",
    config=GPT_specs["GPT-39B"],
    estimator_C={
      'param': 1536.859375, 'activation': 16.0, 'param_optim_mem': 6147.4375, 'activation_mem_np': 80.0, 'activation_mem_p': 727.4720458984375, 'comp': 4.5160591245058805, 'optimizer': 16.56071304814864, 'backward': 1.6589479376866796, 'comp_commu_overlap': 1.0
    },
    rubick_perf_params={
      'k_bwd': 1.0, 'k_sync': 1.0771377039397656, 'k_opt': 0.019934513525821883, 'k_os_offload': 1.0, 'k_const': 1e-08, 'k_off': 1.0, 'k_swap': 1.0
    },
    sia_perf_params={
      'alpha_c': 0.016222343018331052, 'beta_c': 0.10458121076683953, 'alpha_n': 0.3655966778668326, 'beta_n': 0.008101533462094957, 'alpha_r': 1e-08, 'beta_r': 1e-08, 'gamma': 1.491442372461421
    }
  )

ModelConfig_GPT_specs["GPT-76B"] = \
  ModelConfig.from_huggingface(
    model_name="GPT-76B",
    config=GPT_specs["GPT-76B"],
    estimator_C={
      'param': 2400.25390625, 'activation': 20.0, 'param_optim_mem': 9601.015625, 'activation_mem_np': 100.0, 'activation_mem_p': 918.2673950195312, 'comp': 4.529631545850998, 'optimizer': 22.288596813643604, 'backward': 1.7733768258192146, 'comp_commu_overlap': 1.0
    },
    rubick_perf_params={
      'k_bwd': 2.3822122401810617, 'k_sync': 1.0, 'k_opt': 0.02341872911270301, 'k_os_offload': 1.0, 'k_const': 0.0036442995129990236, 'k_off': 1.0, 'k_swap': 1.0
    },
    sia_perf_params={
      'alpha_c': 0.21807601360580986, 'beta_c': 0.04382569231571118, 'alpha_n': 0.5641020945464267, 'beta_n': 1e-08, 'alpha_r': 1e-08, 'beta_r': 1e-08, 'gamma': 1.6704243483328602
    }
  )


ModelConfig_GPT_specs["GPT-175B"] = \
  ModelConfig.from_huggingface(
    model_name="GPT-175B",
    config=GPT_specs["GPT-175B"],
    estimator_C={
      'param': 3456.3046875, 'activation': 24.0, 'param_optim_mem': 13825.21875, 'activation_mem_np': 120.0, 'activation_mem_p': 1100.2673950195312, 'comp': 6.149652008537368, 'optimizer': 31.675269723912287, 'backward': 1.8171811681709664, 'comp_commu_overlap': 1.0
    },
    rubick_perf_params={
      'k_bwd': 2.9490540469196644, 'k_sync': 1.6996669313658173, 'k_opt': 0.012322019191701417, 'k_os_offload': 1.0, 'k_const': 0.00012097929446862094, 'k_off': 1.0, 'k_swap': 1.0
    },
    sia_perf_params={
      'alpha_c': 0.3198479430795331, 'beta_c': 0.0022949449132071936, 'alpha_n': 0.41280565736278574, 'beta_n': 0.005933856165112311, 'alpha_r': 1e-08, 'beta_r': 1e-08, 'gamma': 1.0
    }
  )


ModelStrategy_GPT_specs: Dict[str, ModelStrategy] = {
  "GPT-350M": 
    ModelStrategy(
      model_config=ModelConfig_GPT_specs["GPT-350M"],
      acc_type=default_acc_type,
      global_batch_size=32,
      num_micro_batch=2,
      pp_dp_tp=(1, 1, 1),
      iter_time=0.605
    ),
  "GPT-1.3B":
    ModelStrategy(
      model_config=ModelConfig_GPT_specs["GPT-1.3B"],
      acc_type=default_acc_type,
      global_batch_size=64,
      num_micro_batch=2,
      pp_dp_tp=(1, 4, 1),
      iter_time=0.697
    ),
  "GPT-2.6B":
    ModelStrategy(
      model_config=ModelConfig_GPT_specs["GPT-2.6B"],
      acc_type=default_acc_type,
      global_batch_size=32,
      num_micro_batch=2,
      pp_dp_tp=(1, 4, 1),
      iter_time=0.623
    ),
  "GPT-6.7B":
    ModelStrategy(
      model_config=ModelConfig_GPT_specs["GPT-6.7B"],
      acc_type=default_acc_type,
      global_batch_size=32,
      num_micro_batch=2,
      pp_dp_tp=(1, 4, 2),
      iter_time=0.590
    ),
  "GPT-15B":
    ModelStrategy(
      model_config=ModelConfig_GPT_specs["GPT-15B"],
      acc_type=default_acc_type,
      global_batch_size=48,
      num_micro_batch=4,
      pp_dp_tp=(1, 4, 4),
      iter_time=1.159
    ) if not testbed_no_ib else ModelStrategy(
      model_config=ModelConfig_GPT_specs["GPT-15B"],
      acc_type=default_acc_type,
      global_batch_size=48,
      num_micro_batch=6,
      pp_dp_tp=(2, 4, 2),
      iter_time=1.201
    ),
  "GPT-39B":
    ModelStrategy(
      model_config=ModelConfig_GPT_specs["GPT-39B"],
      acc_type=default_acc_type,
      global_batch_size=64,
      num_micro_batch=8,
      pp_dp_tp=(2, 4, 4),
      iter_time=1.651
    ),
  "GPT-76B":
    ModelStrategy(
      model_config=ModelConfig_GPT_specs["GPT-76B"],
      acc_type=default_acc_type,
      global_batch_size=128,
      num_micro_batch=16,
      pp_dp_tp=(2, 4, 8),
      iter_time=2.994
    ),
  "GPT-175B":
    ModelStrategy(
      model_config=ModelConfig_GPT_specs["GPT-175B"],
      acc_type=default_acc_type,
      global_batch_size=256,
      num_micro_batch=32,
      pp_dp_tp=(4, 8, 8),
      iter_time=3.2571
    ),
}
