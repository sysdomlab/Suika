# Metis/model/activation_parameter.py
# Copyright 2024 Samsung Electronics Co., Ltd. All Rights Reserved

"""
    @adaptation: model_volume used in cost_estimator.py is GPTActivationAndParam class instance in original Metis. We implement an abstract class ActivationAndParam to accommodate general model.
    two methods are needed:
    1. get_activation_size(self, layer_id, batch_size, tp_deg)
    2. get_parameter_size_by_stage(self, tp_deg, start_layer_id, end_layer_id)    
    
    @adaptation: remove first embedding and last decoding layer, assume fully homogeneous transformer model
"""

from Suika.external.Metis.utils import MetisModelConfig
from typing import Dict

DTYPE_BYTES_PER_ELEMENT = 2 # FP16

class ActivationAndParam:
    def __init__(self):
        pass
    
    # get activation size in bytes of one layer
    #   for pp cross-stage transfer
    def get_activation_size(self, layer_id, batch_size, tp_deg):
        raise NotImplementedError("abstract method get_activation_size not implemented")
    
    # get parameter size in bytes of layers [start_layer_id, end_layer_id)
    #   for dp grad transfer
    def get_parameter_size_by_stage(self, tp_deg, start_layer_id, end_layer_id):
        raise NotImplementedError("abstract method get_parameter_size_by_stage not implemented")
    
class GPTActivationAndParam(ActivationAndParam):
    
    def __init__(self, model_config: MetisModelConfig, model_params: Dict):
        # model_params is from profile data field "parameters"
        """
        Initialize the GPTActivationAndParam
        """
        super().__init__()
        self.hidden_size = model_config.hidden_size
        self.sequence_length = model_config.sequence_length
        self.num_layers = model_config.num_layers
        self.transformer_params_per_layer = float(model_params[0])

    #   original Metis return num_element but used as volume in bytes
    #       perhaps it's for pp communication that's too marginal to identify the bug.
    #   our implementation return volume in bytes directly
    
    def get_activation_size(self, layer_id, batch_size, tp_deg):
        return batch_size * self.sequence_length * self.hidden_size * DTYPE_BYTES_PER_ELEMENT
    
    def get_parameter_size_by_stage(self, tp_deg, start_layer_id, end_layer_id):
        return (end_layer_id - start_layer_id) * self.transformer_params_per_layer / tp_deg
