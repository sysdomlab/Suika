# Metis/model/cluster_bandwidth.py
# Copyright 2024 Samsung Electronics Co., Ltd. All Rights Reserved

# @adaptation: remove homogeneous cluster bandwidth as not needed
# @adaptation: refactor get_bandwidth methods to use ResourceManager directly (to GB/s or roughly MB/ms)
# @adaptation: modify device_placement / _sorted_device_types_by_node_sequence to allow different acc_num per node;
"""
    @adapation
        allowing different acc_num per node leads to a bug in rank_node_map, 
        where after reordering ranks according to node_sequence, the rank_node_map becomes invalid (misaligned)
        
        to fix this, we need to remap rank_node_map with respect to node_sequence
"""

from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict, Counter
from typing import Tuple, Dict, List, Optional
# from numpy.typing import NDArray

from Suika.external.Metis.gpu_cluster import GPUCluster
from Suika.external.Metis.search_space.plan import InterStagePlan

class ClusterBandwidth(ABC):
    def __init__(self, gpu_cluster: GPUCluster):
        self.gpu_cluster = gpu_cluster
        self.total_devices = gpu_cluster.get_total_num_devices()
        self.rank_map, self.rank_node_map, _ = self._get_device_placement(
                                                        gpu_cluster=gpu_cluster)

    @abstractmethod
    def _get_dp_groups(self, *args):
        pass

    @abstractmethod
    def _get_pp_groups(self, *args):
        pass

    @abstractmethod
    def get_slowest_pp_bandwidth(self, *args):
        pass

    @abstractmethod
    def get_slowest_dp_bandwidth(self, *args):
        pass

    @staticmethod
    def _get_device_placement(gpu_cluster: GPUCluster, node_sequence: Optional[List[str]] = None) -> Tuple[Dict, Dict]:
        rank_map = defaultdict(list)
        rank_node_map = dict()
        rank_node_map_absolute = dict()
        
        nodes = list(gpu_cluster.nodes.items())
        
        if node_sequence:
            idx_node_sequence = {
                node_id: node_sequence.index(node.acc_type_name()) for node_id, node in nodes
            }
            nodes = sorted(nodes, key= lambda x: idx_node_sequence[x[0]])

        # as rank_node_map is only used internally with sorted_device_types, but not externally to query gpu_cluster 
        # remapping node_id to node_id_from0 does not affect the result, while allowing node_id to be discountinuous.
        counter = 0
        for node_id_from0, (node_id, node) in enumerate(nodes):
            for inner_loop in range(node.acc_num):
                rank_map[node_id].append(counter)
                rank_node_map[counter] = node_id_from0
                rank_node_map_absolute[counter] = node_id
                counter += 1


        return rank_map, rank_node_map, rank_node_map_absolute

    def _get_intra_bandwidth(self, device_type: Optional[str] = None) -> int:
        if device_type is None:
            device_type = self.gpu_cluster.get_device_types()[0]
        return self.gpu_cluster.rm._get_intra_bandwidth(device_type) / 1000 # convert to MB/ms

    def _get_inter_bandwidth(self, device_types: Optional[List] = None) -> int:
        if device_types is None:
            device_types = self.gpu_cluster.get_device_types()
            
        # make unique
        device_types = list(set(device_types))
        
        return min(
            [self.gpu_cluster.rm._get_inter_bandwidth(type1, type2) 
                for type1 in device_types for type2 in device_types]
        ) / 1000 # convert to MB/ms


class HetClusterBandwidth(ClusterBandwidth):
    def __init__(self, gpu_cluster: GPUCluster, plan: InterStagePlan):
        super().__init__(gpu_cluster)
        self.plan = plan

        self.node_sequence = plan.node_sequence
        self.device_groups = plan.device_groups
        
        # reorder rank_map and rank_node_map according to node_sequence
        self.rank_map, self.rank_node_map, _ = self._get_device_placement(
            gpu_cluster=gpu_cluster,
            node_sequence=self.node_sequence
        )

    def _get_pp_groups(self, stage_id: int) -> Tuple[List, List]:
        pp_group_ranks = [i for i in range(sum(self.device_groups[:stage_id]), sum(self.device_groups[:stage_id + 2]))]
        pp_group_nodes = [self.rank_node_map[device_rank] for device_rank in pp_group_ranks]
        return pp_group_ranks, pp_group_nodes

    def _get_dp_groups(self, stage_id: int, strategy: Tuple[int, int]) -> List:
        device_ranks = [i for i in range(sum(self.device_groups[:stage_id]), sum(self.device_groups[:stage_id + 1]))]
        dp_deg, tp_deg = strategy

        dp_groups = [[] for _ in  range(dp_deg)]
        for tp_idx in range(tp_deg):
            for dp_idx in range(dp_deg):
                dp_groups[dp_idx].append(device_ranks.pop(0))
        return dp_groups

    def _sorted_device_types_by_node_sequence(self) -> List[str]:
        device_names = [self.gpu_cluster.nodes[node_id].acc_type_name() for node_id in self.gpu_cluster.nodes.keys()]
        node_sequence = [device_type for device_type in self.plan.node_sequence]

        device_type_count = Counter(device_names)
        sorted_device_types = []
        for key in node_sequence:
            sorted_device_types.extend([key] * device_type_count[key])

        # representing one node with one entry of its device_type; if multiple nodes share the same device_type, then multiple entries will be added.
        return sorted_device_types

    def get_slowest_pp_bandwidth(self, stage_id: int) -> int:
        sorted_device_types = self._sorted_device_types_by_node_sequence()
        pp_group_ranks, pp_group_nodes = self._get_pp_groups(stage_id)
        device_types = [sorted_device_types[node_id] for node_id in list(set(pp_group_nodes))]

        # though called device_types, it represents the device types of the nodes; each node corresponds to one entry of device_types
        if len(device_types) == 1:
            return self._get_intra_bandwidth(device_types[0])
        else:
            return self._get_inter_bandwidth(device_types)

    def get_slowest_dp_bandwidth(self, strategy: Tuple[int, int], stage_id: int) -> int:
        slowest_bandwidth = float('inf')
        sorted_device_types = self._sorted_device_types_by_node_sequence()
        dp_groups = self._get_dp_groups(stage_id, strategy)
        for dp_group in dp_groups:
            group_nodes = [self.rank_node_map[device_rank] for device_rank in dp_group]
            device_types = [sorted_device_types[node_id] for node_id in list(set(group_nodes))]

            if len(device_types) == 1:
                bandwidth = self._get_intra_bandwidth(device_types[0])
            else:
                bandwidth = self._get_inter_bandwidth(device_types)

            if bandwidth < slowest_bandwidth:
                slowest_bandwidth = bandwidth

        return slowest_bandwidth
