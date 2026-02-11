# Metis/gpu_cluster.py
# Copyright 2024 Samsung Electronics Co., Ltd. All Rights Reserved

# use Suika.cluster.resource.ResourceManager while retaining the original functionality.
# @adaptation: hostfile/nodefile parsing is replaced by direct use of ResourceManager and node_list
# @adaptation: remove get_intra_bandwidth and get_inter_bandwidth methods; rely on ResourceManager for bandwidth query at cluster_bandwidth.py directly.
# @adaptation: renamed get_num_nodes_by_device_type to get_num_devices_by_type to refleect it's actual functionality
# @adaptation: added get_num_devices_per_node_dict to support variable number of devices per node

from Suika.cluster.resource import ResourceManager, Node

from typing import List, Dict, Set

class GPUCluster:

  rm: ResourceManager
  node_list: List[Node]
  nodes: Dict[int, Node] # Map node index (node_id) to Node object

  # rm: underlying ResourceManager
  # node_list: nodes that are part of the cluster, allocated to this job.
  def __init__(self, rm: ResourceManager, node_list: List[Node]):
    self.rm = rm
    self.node_list = node_list
    # Use node.node_id as the key for the map
    self.nodes = {node.node_id: node for node in node_list if node.node_id is not None}

  def get_num_nodes(self) -> int:
    return len(self.node_list)

  
  def get_num_devices_by_type(self, device_type: str) -> int:
    """Counts the total number of accelerators of a specific type across all nodes."""
    count = 0
    for node in self.node_list:
      if node.acc_type_name() == device_type:
        count += node.acc_num
    return count

  def get_num_devices_per_node(self, node_id: int) -> int:
    """Returns the number of accelerators in a specific node."""
    if node_id not in self.nodes:
      raise ValueError(f"Node ID {node_id} not found in cluster.")
    return self.nodes[node_id].acc_num
  
  def get_num_devices_per_node_dict(self) -> Dict[int, int]:
    return {node.node_id: node.acc_num for node in self.node_list}

  def get_total_num_devices(self) -> int:
    """Returns the total number of accelerators across all nodes."""
    return sum(node.acc_num for node in self.node_list)

  def get_device_types(self) -> List[str]:
    """Returns a list of accelerator type names for each node."""
    return [node.acc_type_name() for node in self.node_list]

  def get_str_device_types(self) -> str:
    """Returns a string representation of unique accelerator types in the cluster."""
    unique_types: Set[str] = {node.acc_type_name() for node in self.node_list}
    return '_'.join(sorted(list(unique_types)))

  def get_device_memory(self, node_id: int) -> int:
    """
    Returns the total memory size of a single GPU within the specified node.
    :param node_id: ID of the node.
    :return: Memory size in MB.
    """
    if node_id not in self.nodes:
      raise ValueError(f"Node ID {node_id} not found in cluster.")
    node = self.nodes[node_id]
    # Get metadata using the node's accelerator type
    acc_meta = self.rm.get_acc_meta(node.acc_type)
    
    # rm already manage the memory size in MB, no scaling needed
    return acc_meta.memory

  def get_device_memory_for_device_type(self, device_type: str) -> int:
    """
    Returns the memory size of the first found accelerator of the specified type.
    :param device_type: The accelerator type name (string).
    :return: Memory size in MB.
    """
    for node in self.node_list:
      if node.acc_type_name() == device_type:
        acc_meta = self.rm.get_acc_meta(node.acc_type)
        return acc_meta.memory
    raise ValueError(f"Device type '{device_type}' not found in the cluster.")

  # def get_intra_bandwidth(self, node_id: int) -> int:
  #   if node_id not in self.nodes:
  #     raise ValueError(f"Node ID {node_id} not found in cluster.")
  #   acc_type = self.nodes[node_id].acc_type_name()
  #   return self.rm._get_intra_bandwidth(acc_type) * 1024 * 1024
  
  # def get_inter_bandwidth(self, node_id: int) -> int:
  #   if node_id not in self.nodes:
  #     raise ValueError(f"Node ID {node_id} not found in cluster.")
  #   acc_type = self.nodes[node_id].acc_type_name()
  #   return self.rm._get_inter_bandwidth(acc_type, acc_type) * 1024 * 1024


