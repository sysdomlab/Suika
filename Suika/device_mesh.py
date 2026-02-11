from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from Suika.utils import is_power_of_2


# DeviceMesh1D/2D/3D
#
# Dimension:  PP(0)-DP(1)-TP(2)
# Index:      p     i     j
# TP Group: process group to sync input/output for TP 
#           for fixed s, i, [(s, i, j) for i in range(tp_deg)] is a TP group
#           [vertically slice a DeviceMesh2D]
# DP Group: process group to sync grad for DP to apply collective primitives
#           for fixed s, j, [(s, i, j) for i in range(dp_deg)] is a DP group (considered virtual replication)
#           [horizontally slice a DeviceMesh2D]
# 
# PP Group: the entire world for the pipeline, namely all device in DeviceMesh3D
#
# Device:    actually rank(int)


# note: 
#   As we allow different TP degrees = 2^{k_i} within a stage, a single device in a smaller TP group i will be (virtually) replicated to 2^{k_max - k_i} DP groups.
#   This ensures that each TP group intersects with exactly the same number of DP groups, forming a rectangular communication partition for a stage.
#
#   for example:
#       DeviceMesh2D = [[0, 1], [2, 3, 4, 5]]
#       TP Group = [0, 1], [2, 3, 4, 5]
#       DP Group = [0, 2], [0, 3], [1, 4], [1, 5] where 0, 1 are virtually replicated


# note:
#   The reason why we do not define PP group as Megatron does is that, we allow arbitrarily different DP degrees across stages, thus aligned communication is not possbile. We leverage aync NCCL p2p communication for cross stage transfer.

"""
DeviceMesh1D
"""

# RawDeviceMesh means the distributed group will not be initialized
class RawDeviceMesh1D:
  mesh: List[int]
  
  def __init__(self, mesh1D: List[int], validate = False):
    if isinstance(mesh1D, RawDeviceMesh1D):
      # Copy attributes from mesh1D to self
      self.__dict__.update(mesh1D.__dict__)
      return
    
    self.mesh = list(map(int, mesh1D)) if mesh1D is not None else None
    
    if validate and not self.validate():
      raise RuntimeError("DeviceMesh1D: Validation failed")
  
  # coord
  def get_coord(self, rank: int) -> Union[int, None]:
    if rank in self.mesh:
      return self.mesh.index(rank)
    else:
      return None
    
  def get_rank(self, coord: Tuple[int]) -> Union[int, None]:
    if not isinstance(coord, int):
      coord = coord[0]
    if coord < len(self.mesh):
      return self.mesh[coord]
    else:
      return None
    
  # validate
  def validate(self) -> bool:
    visited = set()
    for rank in self.mesh:
      if not isinstance(rank, int):
        raise RuntimeError(f"DeviceMesh1D: Device not a rank (int) : {self}")
        return False
      if rank in visited:
        raise RuntimeError(f"DeviceMesh1D: Detected duplication : {self}")
        return False
      visited.add(rank)
    return True  
    
  # utils
  def flatten(self) -> List[int]:
    return self.mesh
  
  def size(self) -> int:
    return len(self.mesh)

  def tp_deg(self) -> int:
    return len(self.mesh)
  
  def to_list(self) -> List[int]:
    return self.mesh
  
  # overload
  def __getitem__(self, idx):
    return self.mesh[idx]
  
  def __len__(self) -> int:
    return len(self.mesh)

  def __str__(self) -> str:
    return str(self.to_list())
  
  # init_dist
  def get_dist_initialized(self) -> "DeviceMesh1D":
    return DeviceMesh1D(self)
  
  

class DeviceMesh1D(RawDeviceMesh1D): # for TP group
  mesh: List[int]
  group: dist.ProcessGroup
  
  def __init__(self, 
               mesh1D: Union[List[int], RawDeviceMesh1D],
               validate = False,
              ):
    
    if isinstance(mesh1D, DeviceMesh1D):
      # Copy attributes from mesh1D to self
      self.__dict__.update(mesh1D.__dict__)
      return

    super().__init__(mesh1D = None, validate = False)
    
    if isinstance(mesh1D, RawDeviceMesh1D):
      mesh1D = mesh1D.mesh

    self.mesh = list(map(int, mesh1D))
    
    self.group = dist.new_group(self.mesh)

    if validate and not self.validate():
      raise RuntimeError("DeviceMesh1D: Validation failed")

  # group
  def get_tp_group(self) -> dist.ProcessGroup:
    return self.group
      
  # def get_rank_by_group(self, dp_grp_id: int) -> int:
  #   pass
  # won't be used
    


"""
DeviceMesh2D
"""

class RawDeviceMesh2D:
  mesh: List[RawDeviceMesh1D]

  def __init__(self,
               mesh2D: Union[List[RawDeviceMesh1D], List[List[int]]],
               validate = False,
              ):

    if isinstance(mesh2D, RawDeviceMesh2D):
      # Copy attributes from mesh2D to self
      self.__dict__.update(mesh2D.__dict__)
      return
    
    self.mesh = list(map(RawDeviceMesh1D, mesh2D)) if mesh2D is not None else None
    
    if validate and not self.validate():
      raise RuntimeError("DeviceMesh2D: Validation failed")
    
  # coord
  def get_coord(self, rank: int) -> Union[Tuple[int, int], None]:
    for mesh1D in self.mesh:
      coord = mesh1D.get_coord(rank)
      if coord is not None:
        return (self.mesh.index(mesh1D), coord)
    return None
    
  def get_rank(self, coord: Tuple[int, int]) -> Union[int, None]:
    first, *rest = coord
    if first < len(self.mesh):
      return self.mesh[first].get_rank(rest)
    else:
      return None
    
  # validate
  def validate(self):
    
    
    for mesh1D in self.mesh:
      
      if not mesh1D.validate():
        raise RuntimeError(f"DeviceMesh2D: Invalid mesh1D : {self}")
        return False
      
      if not is_power_of_2(mesh1D.tp_deg()):
        raise RuntimeError(f"DeviceMesh2D: TP degree should be power of 2 : {self}")
        return False
    
    visited = set()
    for rank in self.flatten():
      if rank in visited:
        raise RuntimeError(f"DeviceMesh2D: Detected duplication : {self}")
        return False
      visited.add(rank)
      
    return True
  
  # utils
  def flatten(self) -> List[int]:
    flattened = []
    for mesh1D in self.mesh:
      flattened.extend(mesh1D.flatten())
    return flattened
  
  def size(self) -> List[int]:
    return list(map(lambda x: x.size(), self.mesh))  
  
  def tp_deg(self):
    return len(self.dp_groups)
  
  def dp_deg(self):
    return len(self.mesh)
  
  def to_list(self) -> List[List[int]]:
    return [mesh1D.to_list() for mesh1D in self.mesh]
    
  # overload
  def __getitem__(self, idx):
    return self.mesh[idx]
  
  def __len__(self) -> int:
    return len(self.mesh)
  
  def __str__(self) -> str:
    return str(self.to_list())
  
  # init dist
  def get_dist_initialized(self) -> "DeviceMesh2D":
    return DeviceMesh2D(self)
  
    
class DeviceMesh2D(RawDeviceMesh2D): # for DP group
  
  mesh: List[DeviceMesh1D]
  dp_groups: List[dist.ProcessGroup]
  group: dist.ProcessGroup # the entire DP group
  
  def __init__(self, 
               mesh2D: Union[List[DeviceMesh1D], List[List[int]], RawDeviceMesh2D], 
               validate = False
              ):

    if isinstance(mesh2D, DeviceMesh2D):
      # Copy attributes from mesh2D to self
      self.__dict__.update(mesh2D.__dict__)
      return

    super().__init__(mesh2D = None, validate = False)
    
    if isinstance(mesh2D, RawDeviceMesh2D):
      mesh2D = mesh2D.mesh
    
    self.mesh = list(map(DeviceMesh1D, mesh2D))
    
    max_TP_deg = max(list(map(lambda x: x.tp_deg(), self.mesh)))
    self.dp_groups = [None] * max_TP_deg
    self._build_dp_groups()
    
    self.group = dist.new_group(self.flatten())
    
    if validate and not self.validate():
      raise RuntimeError("DeviceMesh2D: Validation failed")
  
  ## rank-group querys
  # notations:
  #   coord: (i, j) of (dim 0, dim 1)
  #   tp_grp_id: which tp group (DP dim), same as i actually
  #   dp_grp_id: which dp group (TP dim)
  #   rank: global rank
  def get_group(self):
    return self.group
  
  def get_tp_group(self, i: int) -> dist.ProcessGroup:
    return self.mesh[i].get_tp_group()
  
  def get_tp_group_by_rank(self, rank: int) -> dist.ProcessGroup:
    coord = self.get_coord(rank)
    if coord is None:
      raise RuntimeError("DeviceMesh2D: Invalid rank")
    i, _ = coord
    return self.get_tp_group(i)
  
  def get_dp_group(self, j: int) -> dist.ProcessGroup:
    return self.dp_groups[j]
  
  def get_dp_groups(self, i: int, j: int) -> List[dist.ProcessGroup]:
    max_TP_deg = self.tp_deg()
    cur_TP_deg = self.mesh[i].tp_deg()
    
    begin = j * max_TP_deg // cur_TP_deg
    end = (j + 1) * max_TP_deg // cur_TP_deg
    
    return self.dp_groups[begin:end]
  
  def get_dp_groups_by_rank(self, rank: int) -> List[dist.ProcessGroup]:
    coord = self.get_coord(rank)
    if coord is None:
      raise RuntimeError("DeviceMesh2D: Invalid rank")
    i, j = coord
    return self.get_dp_groups(i, j)
  
  def get_coord_by_group(self, tp_grp_id: int, dp_grp_id: int) -> Tuple[int, int]:
    if tp_grp_id >= self.dp_deg() or dp_grp_id >= self.tp_deg():
      raise RuntimeError("DeviceMesh2D: Invalid group id")
    return (tp_grp_id, dp_grp_id * self.mesh[tp_grp_id].tp_deg() // self.tp_deg())
  
  def get_rank_by_group(self, tp_grp_id: int, dp_grp_id: int) -> int:
    if tp_grp_id >= self.dp_deg() or dp_grp_id >= self.tp_deg():
      raise RuntimeError("DeviceMesh2D: Invalid group id")
    i, j = self.get_coord_by_group(tp_grp_id, dp_grp_id)
    return self.mesh[i][j]
  
  # initialize dp group
  def _build_dp_groups(self):
          
    # assume we have max_TP_deg = 4 with cur_TP_deg = 2 (GPU0, GPU1)
    #   we should assign GPU0 to DP group 0, 1, GPU1 to DP group 2, 3
    #
    #   but how to provide sync boundary if a physic GPU is virtualized to 2 DP groups?
    #   seems NCCL provides that isolation between ProcessGroups.
    for dp_group_id in range(self.tp_deg()):
      device_lists = []
      for tp_group_id in range(self.dp_deg()):
        # locate the proper device in TP group to add to current dp group
        rank = self.get_rank_by_group(tp_grp_id=tp_group_id, dp_grp_id=dp_group_id) 
                
        device_lists.append(rank)
      
      self.dp_groups[dp_group_id] = dist.new_group(device_lists)
        
        
"""
DeviceMesh3D
"""

class RawDeviceMesh3D:
  mesh: List[RawDeviceMesh2D]
  
  def __init__(self,
               mesh3D: Union[List[RawDeviceMesh2D], List[List[List[int]]]],
               validate = False,
              ):
    if isinstance(mesh3D, RawDeviceMesh3D):
      # Copy attributes from mesh3D to self
      self.__dict__.update(mesh3D.__dict__)
      return
    
    self.mesh = list(map(RawDeviceMesh2D, mesh3D)) if mesh3D is not None else None
    
    if validate and not self.validate():
      raise RuntimeError("DeviceMesh3D: Validation failed")
  
  # coord
  def get_coord(self, rank: int) -> Union[Tuple[int, int, int], None]:
    for mesh2D in self.mesh:
      coord = mesh2D.get_coord(rank)
      if coord is not None:
        return (self.mesh.index(mesh2D), coord[0], coord[1])
    return None
  
  def get_rank(self, coord: Tuple[int, int, int]) -> Union[int, None]:
    first, *rest = coord
    if first < len(self.mesh):
      return self.mesh[first].get_rank(rest)
    else:
      return None
  
  # validate
  def validate(self):
    for mesh2D in self.mesh:
      if not mesh2D.validate():
        raise RuntimeError(f"DeviceMesh3D: Invalid mesh2D : {self}")
        return False
    
    visited = set()
    for rank in self.flatten():
      if rank in visited:
        raise RuntimeError(f"DeviceMesh3D: Detected duplication : {self}")
        return False
      visited.add(rank)
      
    return True
  
  # utils
  def flatten(self) -> List[int]:
    flattened = []
    for mesh2D in self.mesh:
      flattened.extend(mesh2D.flatten())
    return flattened
  
  def size(self) -> List[List[int]]:
    return list(map(lambda x: x.size(), self.mesh))
  
  def pp_deg(self) -> int:
    return len(self.mesh)
  
  def to_list(self) -> List[List[List[int]]]:
    return [mesh2D.to_list() for mesh2D in self.mesh]
  
  # overload
  def __getitem__(self, idx):
    return self.mesh[idx]

  def __len__(self) -> int:
    return len(self.mesh)
  
  def __str__(self) -> str:
    return str(self.to_list())
  
  # init dist
  def get_dist_initialized(self) -> "DeviceMesh3D":
    return DeviceMesh3D(self)

class DeviceMesh3D(RawDeviceMesh3D): # for PP group
  mesh: List[DeviceMesh2D]
  group: dist.ProcessGroup # essentially the WORLD group in general
  
  def __init__(self, 
                mesh3D: Union[List[DeviceMesh2D], List[List[List[int]]], RawDeviceMesh3D],
              ):
    if isinstance(mesh3D, DeviceMesh3D):
      # Copy attributes from mesh3D to self
      self.__dict__.update(mesh3D.__dict__)
      return
    
    super().__init__(mesh3D = None, validate = False)
    if isinstance(mesh3D, RawDeviceMesh3D):
      mesh3D = mesh3D.mesh
      
    self.mesh = list(map(DeviceMesh2D, mesh3D))
    
    self.group = dist.new_group(self.flatten())
    self.cpu_group = dist.new_group(self.flatten(), backend="gloo")
    
    if not self.validate():
      raise RuntimeError("DeviceMesh3D: Validation failed")
    
  # group
  # won't be used, I suppose
  # def get_tp_group(self, i: int, j: int):
  #   return self.mesh[i].get_tp_group(j)

  # def get_dp_group(self, i: int, j: int):
  #   return self.mesh[i].get_dp_group(j)
  def get_group(self) -> dist.ProcessGroup:
    return self.group
  def get_cpu_group(self) -> dist.ProcessGroup:
    return self.cpu_group

