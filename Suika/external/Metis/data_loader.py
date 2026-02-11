# Metis/data_loader.py
# Copyright 2024 Samsung Electronics Co., Ltd. All Rights Reserved

"""
    @adaptation: 
     to serve multiple model types, we arrange profile_data as 
     
     profile_data
     |-GPT
     | |- DeviceType.A100_80GB_tp1_bs1.json
     | |- DeviceType.A100_80GB_tp1_bs2.json
     | |- ...
     |-llama
     | |- ...
     |-...
     
     the profile_data for specific model type follows naming convention of original Metis
     
     please specify profile_dir as f`profile_data/{model_type}`
     
    also, max_tp, max_bs is returned from load_profile_data_all
    to be used in cost_het_cluster.py
"""


import json
import os
import re
from typing import Dict, Union, List, Tuple

class ProfileDataLoader:
    def __init__(self, profile_dir: str):
        self.profile_dir = profile_dir
        # Find a list of files that contain profile data
        self.profile_data_list = [fname for fname in os.listdir(profile_dir) if fname.endswith('.json')]

    def _get_model_profile_data(self, raw_data: Dict[str, Dict]) \
            -> Dict[str, Dict[str, Union[int, float, List]]]:
        model_profile_data = dict()
        model_profile_data['optimizer_time'] = raw_data['execution_time']['optimizer_time_ms'] * 2
        num_layers = len(raw_data['execution_time']['layer_compute_total_ms'])
        model_profile_data['num_layers'] = num_layers
        model_profile_data['batch_generator'] = raw_data['execution_time']['batch_generator_time_ms']
        model_profile_data['parameters'] = raw_data['model']['parameters']['parameters_per_layer_bytes']
        return model_profile_data

    def _get_device_type_specific_profile_data(self, raw_data: Dict[str, Dict[str, Union[int, float, List]]])\
            -> Dict[str, Dict[str, Union[int, float, List]]]:
        profile_data = dict()
        profile_data["time"] = dict()
        layer_computes = raw_data['execution_time']['layer_compute_total_ms']
        layer_compute_times = [layer_compute for layer_compute in layer_computes]
        profile_data["time"]["layer-computes"] = layer_compute_times
        forward_backward_time = raw_data['execution_time']['forward_backward_time_ms']
        profile_data["time"]["fb_sync"] = forward_backward_time - sum(layer_compute_times)
        profile_data['memory'] = raw_data['execution_memory']['layer_memory_total_mb']

        return profile_data

    def load_profile_data_all(self) -> Tuple[Dict, List]:
        profile_data = dict()
        device_types = []
        
        max_tp = 0
        max_bs = 0
        
        for profile_data_path in self.profile_data_list:
            device_type = re.search(r"DeviceType\.(\w+?)_", profile_data_path).group(1)
            if f'DeviceType.{device_type}' not in profile_data.keys():
                profile_data[f'DeviceType.{device_type}'] = dict()
                device_types.append(device_type)

            tp = re.search(r"tp(\d+)", profile_data_path).group(1)
            max_tp = max(max_tp, int(tp))
            bs = re.search(r"bs(\d+)", profile_data_path).group(1)
            max_bs = max(max_bs, int(bs))

            with open(f'{self.profile_dir}/{profile_data_path}', 'r') as content:
                raw_profile_data = json.loads(content.read())

                if "model" not in profile_data.keys():
                    model_profile_data = self._get_model_profile_data(raw_profile_data)
                    profile_data["model"] = model_profile_data

                profile_data[f"DeviceType.{device_type}"][f"tp{tp}_bs{bs}"] = \
                    self._get_device_type_specific_profile_data(raw_profile_data)

        return profile_data, device_types, max_tp, max_bs
