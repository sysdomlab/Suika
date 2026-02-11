# Metis/utils.py
# Copyright 2024 Samsung Electronics Co., Ltd. All Rights Reserved

# @adaptation: DeviceType enum from Metis utils is replaced by string comparison
# @adaptation: add conversion from transformer's GPT2Config to MetisModelConfig

import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Union, List

from Suika.models.utils import ModelConfig

def parse_hostfile(file_path: str) -> Dict[int, Dict[str, Union[str, int]]]:
    num_node, hostfile_info = 0, dict()
    with open(file_path, 'rt') as hostfile:
        line = hostfile.readline()
        while line:
            splitted_data = line.split(' ')
            ip = splitted_data[0]
            num_device = int(splitted_data[1][6:7])

            hostfile_info[num_node] = dict()
            hostfile_info[num_node]["ip"] = ip
            hostfile_info[num_node]["num_device"] = num_device

            line = hostfile.readline()
            num_node += 1

    return hostfile_info


def parse_nodefile(file_path: str) -> Dict[str, Dict[str, Union[str, int]]]:
    with open(file_path, 'r') as content:
        clusters = json.loads(content.read())

    return clusters


def factor(N: int, upper: int = None, lower: int = None) -> List:
    if upper is None:
        upper = N

    ret = []
    for i in range(1, upper + 1):
        if N % i == 0:
            if lower is None or i >= lower:
                ret.append(i)
    return ret


class DeviceType(str):

    @staticmethod
    def from_string(s: str) -> 'DeviceType':
      return s.upper()

@dataclass
class ResourceConfig:
    device_type: DeviceType
    inter_bw: int
    intra_bw: int
    num_nodes: int
    num_devices: int
    total_devices: int
    device_memory: int

@dataclass
class MetisModelConfig:
    model_name: str
    num_layers: int
    hidden_size: int
    sequence_length: int
    hidden_size: int

    @staticmethod
    def from_SuikaConfig(suika_config: ModelConfig):
        
        model_name = suika_config.model_name
        num_layers = suika_config.n_layer
        hidden_size = suika_config.n_embd
        sequence_length = suika_config.n_positions
        return MetisModelConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            sequence_length=sequence_length,
            model_name=model_name
        )

@dataclass
class GPUNode:
    device_type: DeviceType
    num_devices: int

import logging
from Suika.utils import get_logger
metis_logger = get_logger("metis", level=logging.INFO, console_level=logging.ERROR) # if set to INFO/DEBUG, file too large at simulation

def metis_print(s: str):
    """
    Print function for Metis, using the metis_logger.
    """
    pass
    # metis_logger.debug(s)
