# 1. Accelerator (namely, a GPU in our case)
# 2. Node
# 3. cluster-wide nodes management
import os
import math
import copy
from typing import Callable, Iterable, Optional, List, Dict, Tuple, Union

from Suika.utils import flatten_mesh

"""
Default Resource Constants

We make little assumption for supported resource setup, but for convenience, we provide some default values.

These values are not used except for predefined execution plans for experiment benchmark / external baseline.
"""
default_testbed = os.getenv("SUIKA_TESTBED", "cluster-a").lower()
testbed_half_nics: bool = default_testbed == "cluster-b"
testbed_no_ib: bool = default_testbed == "cluster-c"

cross_rack_discount = 0.5
node_per_rack = 16

if default_testbed == "debug":
  # print("Testbed: debug")
  default_acc_type = "V100"
  default_acc_TFLOPS = 31.4 # FP16
  default_acc_memory = 16 * 1000 # MB
  default_intra_bandwidth = 6.5 * 1000 # MB/s
  default_intra_bandwidth_saturate = 64
  default_inter_bandwidth = 6.5 * 1000 # MB/s
  default_n_acc_per_node = 4
  default_n_node = 2
  
  default_rack_id_map = None
elif default_testbed == "cluster-a":
  # normal
  # print("Testbed: cluster-a")
  default_acc_type = "H100"
  default_acc_TFLOPS = 989.4
  default_acc_memory = 80 * 1000
  default_intra_bandwidth = 300 * 1000
  default_intra_bandwidth_saturate = 2048
  default_inter_bandwidth = 50 * 1000
  default_n_acc_per_node = 8
  default_n_node = 8
  
  default_rack_id_map = None
elif default_testbed == "cluster-b":
  # half NICs
  print("Testbed: cluster-b")
  default_acc_type = "H100"
  default_acc_TFLOPS = 989.4
  default_acc_memory = 80 * 1000
  default_intra_bandwidth = 300 * 1000
  default_intra_bandwidth_saturate = 2048
  # default_inter_bandwidth = 25 * 1000
  default_inter_bandwidth = 50 * 1000 # single NIC effective bdw is still 400 Gbps
  default_n_acc_per_node = 8
  default_n_node = 8
  
  default_rack_id_map = None
elif default_testbed == "cluster-c":
  # no IB
  print("Testbed: cluster-c")
  default_acc_type = "H100"
  default_acc_TFLOPS = 989.4
  default_acc_memory = 80 * 1000
  default_intra_bandwidth = 300 * 1000
  default_intra_bandwidth_saturate = 2048
  default_inter_bandwidth = 1.25 * 1000
  default_n_acc_per_node = 8
  default_n_node = 8
  
  default_rack_id_map = None
elif default_testbed == "cluster-d":
  # simulation
  print("Testbed: cluster-d")
  default_acc_type = "H100"
  default_acc_TFLOPS = 989.4
  default_acc_memory = 80 * 1000
  default_intra_bandwidth = 300 * 1000
  default_intra_bandwidth_saturate = 2048
  default_inter_bandwidth = 50 * 1000
  default_n_acc_per_node = 8
  default_n_node = int(os.getenv("SUIKA_TESTBED_N_NODE", str(1024 // 8)))
  
  def rack_id_map_lambda(acc: "Accelerator"):
    if isinstance(acc, str):
      return 0
    else:
      return acc.node_id // node_per_rack  # 16 nodes per rack
  default_rack_id_map = rack_id_map_lambda # since e2e experiment data is pickpled wihtout rack_id_map in rm, so we have to define it as global as a workaround, so that the pickled data is still loadable.
else:
  raise ValueError(f"Unknown default testbed {default_testbed}")

"""
AcceleratorMeta
"""
class AcceleratorMeta():
  type_name: str
  TFLOPS: float # FP16
  memory: int # in MB
  
  def __init__(self, type_name:str, TFLOPS: float, memory: int):
    self.type_name = type_name
    self.TFLOPS = TFLOPS
    self.memory = memory
  
  def __str__(self):
    return f"({self.type_name}, {self.TFLOPS} TFLOPS, {self.memory} MB)"
  
  
"""
Accelerator:
  1. accelerator type (meta)
  2. device id = (node_id, local_id)
"""
class Accelerator():
  acc_type: str
  # type meta of GPU
  
  node_id: Optional[int]
  # 0-indexed
  # for None, not set (not recommended)
    
  local_id: Optional[int]
  # 0-indexed
  # intra node accelerator id
  # for None, not set (will be infered later)
  
  def __init__(self, 
               acc_type: str, 
               node_id: Optional[int] = None, 
               local_id: Optional[int] = None # intra node id
               ):
    self.acc_type = acc_type
    self.node_id = node_id
    self.local_id = local_id
    
  # (node_id, local_id)
  def get_coord(self) -> Tuple[Optional[int], Optional[int]]:
    return (self.node_id, self.local_id)
  
  def acc_type_name(self):
    return self.acc_type if isinstance(self.acc_type, str) else self.acc_type.type_name
    
  def __str__(self):
    return f"Accelerator(acc_type={self.acc_type_name()}, id={(self.node_id, self.local_id)})"
  def __repr__(self):
    return self.__str__()
  
  def __hash__(self):
    return hash((self.acc_type_name(), self.node_id, self.local_id))
  
  def __eq__(self, other):
    if not isinstance(other, Accelerator):
      return False
    return (
      self.node_id == other.node_id and 
      self.local_id == other.local_id and
      self.acc_type_name() == other.acc_type_name())
  
"""
AcceleratorMesh:
"""
AcceleratorMesh = List[List[List[Accelerator]]] # 3D Accelerator mesh
  

"""
Node:
  1. acc_type: accelerator type (meta) 
    - assume all accelerators in the node are of the same type
  2. node_id: node id
  3. acc_num: number of accelerators in the node
"""

class Node():  
  acc_type: str
  # type meta of GPU

  node_id: Optional[int]
  # 0-indexed
  # for None, not set (not recommended)
  
  acc_num: int
  # number of accelerators in the node
  
  hostname: Optional[int]
  # ssh hostname, also real host name.
  
  usage: List[bool]
  
  def __init__(self, 
               acc_type: str, 
               node_id: Optional[int], 
               acc_num: int,
               hostname: Optional[str] = None
               ):
    self.acc_type = acc_type
    self.node_id = node_id
    self.acc_num = acc_num
    self.hostname = hostname
    self.usage = [False for i in range(self.acc_num)]
    
  def get_acc_list(self) -> List[Accelerator]:
    return [Accelerator(acc_type=self.acc_type, node_id=self.node_id, local_id=i) for i in range(self.acc_num)]
    
  def get_virtual_acc_list(self, virtual_node_id) -> List[Accelerator]:
    return [Accelerator(acc_type=self.acc_type, node_id=virtual_node_id, local_id=i) for i in range(self.acc_num)]
  
  def get_avail_acc_list(self) -> List[Accelerator]:
    return [Accelerator(acc_type=self.acc_type, node_id=self.node_id, local_id=i) for i in range(self.acc_num) if not self.usage[i]]
  
  def acc_type_name(self):
    return self.acc_type if isinstance(self.acc_type, str) else self.acc_type.type_name
  
  def get_usage(self, local_id: int = None) -> Union[int, bool]:
    if local_id is None:
      return self.usage.count(True)
    else:
      assert 0 <= local_id < self.acc_num, f"local_id {local_id} out of range [{0}, {self.acc_num})"
      return self.usage[local_id]
    
  def get_avail(self):
    return self.acc_num - self.usage.count(True)
  
  def set_usage(self, local_id: int, flag = True):
    assert 0 <= local_id < self.acc_num, f"local_id {local_id} out of range [{0}, {self.acc_num})"
    assert self.usage[local_id] != flag, f"invalid flag, same as current value"
    
    self.usage[local_id] = flag
  
  def reset_usage(self, local_id: int):
    self.set_usage(local_id, False)
  
  def __str__(self):
    return f"Node(\n     id={self.node_id}, acc_type={self.acc_type_name()}, acc_num={self.acc_num}, hostname={self.hostname}\n     usage={self.usage}\n)\n"
  def __repr__(self):
    return self.__str__()
  
  def __hash__(self):
    return hash(self.node_id)
  
  def __eq__(self, other):
    if not isinstance(other, Node):
      return False
    return (
      self.acc_type_name() == other.acc_type_name() and
      self.node_id == other.node_id and
      self.acc_num == other.acc_num and
      self.hostname == other.hostname 
    )
  
"""
NodeMesh
HostnameMesh
"""
HostnameMesh = List[List[List[str]]] # 3D hostname mesh

  
  

"""
ResourceManager per cluster

  manage cluster wide resources and allocate/free
"""
  
class ResourceManager():
  
  """
  Accelerator meta table
  """
  acc_meta_table: Dict[str, AcceleratorMeta] = {}

  """
  Cluster node table
  """
  cluster_node_table: Dict[int, Node]
  
  """
  Cluster network configuration
  
  theoretically, it supports individual node to node bandwidth, but registering that is tedious.
  so we simplify it to B_intra[acc_type], B_inter[acc_type][acc_type]
  """
  bw_intra: Dict[str, float] = {} # intra node bandwidth MB/s
  bw_intra_sat: Dict[str, float] = {} # saturating threshold
  bw_inter: Dict[Tuple[str, str], float] = {} # inter node bandwidth MB/s
    
  def __init__(self):
    self.acc_meta_table = {} # acc_name -> acc_meta
    self.cluster_node_table = {}  # node_id -> node
    self.bw_inter = {}
    self.bw_intra = {}
    self.bw_intra_sat = {}
    
  """
  Accelerator
  """


  """
  @EXPORT
  register accelerator meta in lockup table
  """
  def register_acc_meta(self, type_name: str, TFLOPS: float, memory: int, override = False):
    assert type_name not in self.acc_meta_table or override, f"register_acc_meta: accelerator {type_name} already exists but override set to False"    
    self.acc_meta_table[type_name] = AcceleratorMeta(type_name, TFLOPS, memory)

  """
  @EXPORT
  query accelerator meta in lockup table
  None if not found
  """

  def get_acc_meta(self, acc: Union[str, AcceleratorMeta, Accelerator]) -> AcceleratorMeta:
    if isinstance(acc, Accelerator):
      acc = acc.acc_type
    if isinstance(acc, AcceleratorMeta):
      return acc
    assert acc in self.acc_meta_table, f"get_acc_meta: accelerator {acc} not found"
    return self.acc_meta_table[acc]


  """
  Node
  """


  """
  @EXPORT
  Register the node to the cluster node table

  if node_id is None,
    a minimum spare id will be assigned
  """
  def register_node(self, acc_type: Union[str, AcceleratorMeta], node_id: Optional[int], acc_num: int, hostname: Optional[str] = None, override=False):
    if node_id is None:
      node_id = 0
      while True:
        if node_id not in self.cluster_node_table:
          break
        node_id += 1

    assert node_id not in self.cluster_node_table or override, f"register_node: node id {node_id} already exists but override set to False"

    self.cluster_node_table[node_id] = Node(
      acc_type=acc_type, 
      node_id=node_id, 
      acc_num=acc_num, 
      hostname=hostname
    )

  """
  @EXPORT
  Get the node from the cluster node table
  """
  def get_node(self, node_id: int) -> Node:
    assert node_id in self.cluster_node_table, f"get_node: node {node_id} not found"  
    return self.cluster_node_table[node_id]

  """
  @EXPORT
  Get number of occupied accelerators on node
  """
  def get_node_usage(self, node_id: int, local_id: int = None) -> Union[int, bool]:
    return self.get_node(node_id).get_usage(local_id)
  """
  @EXPORT
  Get number of available accelerators in the 
  """
  def get_node_avail(self, node_id: int) -> int:
    return self.get_node(node_id).get_avail()

  """
  @EXPORT
  Set acclerator (node_id, local_id) usage to flag
  Args:
    1. node_id
    2. local_id: Optional[int], if None, inferred as smallest one that differs flag
    3. flag: bool, value to set, default to True
  Return:
    local_id (after inference)
  """
  def set_node_usage(self, node_id: int, local_id: Optional[int] = None, flag: bool = True, mask: List[Accelerator] = None):
    node = self.get_node(node_id)
    if local_id is None:
      for idx, usage in enumerate(node.usage):
        if usage != flag and \
          (mask is None or Accelerator(node.acc_type, node_id, idx) in mask):
          local_id = idx
          break
      if local_id is None:
        raise RuntimeError(f"set_node_usage: no available local_id found for node {node_id} with flag {flag} under mask {mask}")
    
    node.set_usage(local_id, flag = flag)
    return local_id

  def get_node_usage_dict(self):
    usage_dict = {
      node_id: self.get_node(node_id).usage for node_id in self.cluster_node_table
    }
    return usage_dict

  """
  @EXPORT
  Reset acclerator (node_id, local_id) usage to False
  """
  def reset_node_usage(self, node_id: int, local_id: Optional[int] = None):
    return self.set_node_usage(node_id, local_id, flag = False)
  
  """
  @EXPORT
  Get the list of all nodes in the cluster
  """
  def get_node_list(self) -> List[Node]:
    return list(self.cluster_node_table.values())

  """
  @EXPORT
  Get the list of all available nodes in the cluster; exclude those with available accelerator <= 0
  """
  def get_avail_node_list(self, threshold = 0) -> List[Node]:
    avail_node_list = []
    for node_id, node in self.cluster_node_table.items():
      if node.get_avail() > threshold:
        avail_node_list.append(node)
    return avail_node_list
  
  
  """
  @EXPORT
  
  Get all available accelerators in the cluster
  """
  def get_avail_acc_list(self) -> List[Accelerator]:
    nodes = self.get_avail_node_list()
    
    return [
      acc for node in nodes for acc in node.get_avail_acc_list()
    ]
  
  """
  @EXPORT
  
  Get all available accelerators in the cluster
  """
  def get_avail_acc_mesh2D(self) -> List[List[Accelerator]]:
    nodes = self.get_avail_node_list()
    
    return [
      node.get_avail_acc_list() for node in nodes
    ]
  
  """
  @EXPORT
  
  Get the total number of accelerators in the cluster
  """
  def get_total_acc_num(self) -> int:
    return sum([node.acc_num for node in self.cluster_node_table.values()])
  
  
  
  """
  Network
  """
  
  """
  @EXPORT
  Register the intra-node bandwidth
  unit: MB/s
  """
  def register_bw_intra(self, acc_type: str, bw: float):
    assert acc_type in self.acc_meta_table, f"register_bw_intra: accelerator {acc_type} not found"
    self.bw_intra[acc_type] = bw

  """
  @EXPORT
  Register the intra-node saturate threshold
  """
  def register_bw_intra_sat(self, acc_type: str, sat: float):
    assert acc_type in self.acc_meta_table, f"register_bw_intra_sat: accelerator {acc_type} not found"
    if sat is None:
      return #never register None to sat, for CPP interface simplicity.
    self.bw_intra_sat[acc_type] = sat
    
  """
  @EXPORT
  Register the inter-node bandwidth
  unit: MB/s
  """
  def register_bw_inter(self, acc_type1: str, acc_type2: str, bw: float):
    assert acc_type1 in self.acc_meta_table, f"register_bw_inter: accelerator {acc_type1} not found"
    assert acc_type2 in self.acc_meta_table, f"register_bw_inter: accelerator {acc_type2} not found"
    
    self.bw_inter[(acc_type1, acc_type2)] = bw
    self.bw_inter[(acc_type2, acc_type1)] = bw
  
  def register_bw(self, acc_type_or_types: str, bw: Union[float, Tuple[float, float]]):
    """
    if acc_type_or_types is a 2 acc_type str separated by ','. then inter, else intra
    e.g.:
      "A100,V100"
      "A100"
    if is intra bandwidth, then
      "[bw, sat]" register bw as intra bandwidth, sat as saturate threshold
      "bw", register bw as intra bandwidth, sat as None
    """
    if ',' in acc_type_or_types:
      acc_type1, acc_type2 = map(lambda s: s.strip(), acc_type_or_types.split(','))
      self.register_bw_inter(acc_type1, acc_type2, bw)
    else:
      if isinstance(bw, (list, tuple)):
        assert len(bw) == 2, f"expect [bw, sat], got {bw}"
        bw, sat = bw
      else:
        bw = float(bw)
        sat = None
      acc_type = acc_type_or_types
      self.register_bw_intra(acc_type, bw)
      self.register_bw_intra_sat(acc_type, sat)
  
  """
  @EXPORT
  Get the bandwidth between two accelerator
  """
  def get_bandwidth(self, acc1: Accelerator, acc2: Accelerator) -> float:
    if acc1.node_id == acc2.node_id:
      # intra node bandwidth
      acc1 = acc1.acc_type_name()
      assert acc1 in self.bw_intra, f"get_bandwidth: intra bandwidth for {acc1} not found"
      return self.bw_intra[acc1]
    else:
      # inter node bandwidth
      same_rack = default_rack_id_map(acc1) == default_rack_id_map(acc2) if default_rack_id_map is not None else True
      discount = 1.0 if same_rack else cross_rack_discount
      
      acc1 = acc1.acc_type_name()
      acc2 = acc2.acc_type_name()
      assert (acc1, acc2) in self.bw_inter, f"get_bandwidth: bandwidth between {acc1} and {acc2} not found"

      return self.bw_inter[(acc1, acc2)] * discount
    
  def _get_intra_bandwidth(self, acc_type: str) -> float:
    if isinstance(acc_type, Accelerator):
      acc_type = acc_type.acc_type_name()
    
    assert acc_type in self.bw_intra, f"_get_intra_bandwidth: intra bandwidth for {acc_type} not found"
    return self.bw_intra[acc_type]
  def _get_inter_bandwidth(self, acc_type1: str, acc_type2: str) -> float:
    if isinstance(acc_type1, Accelerator):
      acc_type1 = acc_type1.acc_type_name()
    if isinstance(acc_type2, Accelerator):
      acc_type2 = acc_type2.acc_type_name()
      
    assert (acc_type1, acc_type2) in self.bw_inter, f"_get_inter_bandwidth: bandwidth between {acc_type1} and {acc_type2} not found"
    return self.bw_inter[(acc_type1, acc_type2)]
  
  def get_weak_bandwidth(self, acc1: Accelerator, acc2: Accelerator, volume: float):
    if acc1.node_id != acc2.node_id:
      return self._get_inter_bandwidth(acc1, acc2)

    if volume < 1e-3:
      return 1.0 # avoid log(0), the bw does not matter when volume is 0

    acc = acc1.acc_type_name()
    bw = self._get_intra_bandwidth(acc)
    sat = self.bw_intra_sat.get(acc, None)
    if sat is None:
      return bw
    
    sat_log = math.log2(sat)
    volume_log = math.log2(volume)
    return bw * max(min(volume_log / sat_log, 1.0), 0.2)
    
  
  """
  Scheduling Related
  """
  
  """
    @EXPORT
    Test if the given virtual accelerator mesh can be allocated 
    without modifying the resource manager.
  """
  def test_allocate(self, 
                    v_acc_mesh: AcceleratorMesh,
                    mask: Optional[List[Accelerator]] = None
                    ) -> bool:
    
    return self.allocate(v_acc_mesh=v_acc_mesh, virtual_test=True, mask=mask) is not None
  
  """
  @EXPORT
  Allocate available resources that satisfy given virtual acc_mesh
  
  return None if not satisfiable.
  
  Args:
    1. v_acc_mesh: virtual accelerator mesh
    2. virtual_test: bool, 
        if True, only test the allocation without modifying the resource manager
    3. mask: Optional[List[Accelerator]],
        limit the allocation to the given accelerator mask, default no mask
  Return:
    return the allocated accelerator mesh or None if failed
    
    if virtual_test is True, free will be called before returning
      
  """
  def allocate(self, 
               v_acc_mesh: AcceleratorMesh, 
               virtual_test: bool = False,
               mask: Optional[List[Accelerator]] = None
               ) \
    -> Optional[AcceleratorMesh]:
    
    """ process virtual part """
    acc_num_dict = {} # key: node_id, value: acc_num needed
    acc_type_dict = {}
    acc_type_set = set()
    for acc in flatten_mesh(v_acc_mesh):
      acc: Accelerator
      
      node_id = acc.node_id
      acc_type = acc.acc_type_name()
      acc_num_dict[node_id] = acc_num_dict.get(node_id, 0) + 1
      acc_type_dict[node_id] = acc_type
      acc_type_set.add(acc_type)
    
    # split to lists according to acc_type
    required_node_dict = { acc_type: [] for acc_type in acc_type_set} 
    # key: acc_type, value list of (required_acc_num, node_id)
    for node_id in acc_num_dict.keys():
      acc_num = acc_num_dict[node_id]
      acc_type = acc_type_dict[node_id]
      
      required_node_dict[acc_type].append([acc_num, node_id])

    """ process physical part """
    # get physical avail nodes
    avail_node_list = self.get_avail_node_list()
    avail_node_dict = { acc_type: [] for acc_type in acc_type_set }
    # key: acc_type, value list of (available_acc_num, node_id)
    
    # get available devices on each node with regards to mask
    if mask is not None:
      mask = flatten_mesh(mask)
      for acc in mask:
        assert not self.get_node(acc.node_id).get_usage(acc.local_id), \
          f"allocate: accelerator {acc} in mask is already allocated, cannot be used for allocation"
      masked_max_node_avail = { acc.node_id: 0 for acc in mask }
      for acc in mask:
        masked_max_node_avail[acc.node_id] += 1

    def get_avail(node: Node):
      return node.get_avail() if mask is None else \
            min(node.get_avail(), masked_max_node_avail.get(node.node_id, 0))
    
    for node in avail_node_list:
      acc_type = node.acc_type_name()
      if acc_type not in acc_type_set: # plan do not require this type of GPU
        continue
      node_num_avail_acc = get_avail(node)
      if node_num_avail_acc <= 0: # no available GPU
        continue
      avail_node_dict[acc_type].append([node_num_avail_acc, node.node_id])
      
    """ solve with minimal fit """
    
    """
      remark: this implement is brutal, which takes O(n) to search for one round
              a binary search on BST is reasonable to reduce it to O(log n)
              but considering simplicity and low frequency to call allocate(), 
                we leave it as it is.

    """
    def find_minimal_fit(required: Tuple[int, int], avail_list: List[Tuple[int, int]]) -> int:
        best_val, best_idx = -1, -1
        for idx, avail in enumerate(avail_list):
          if avail[0] >= required[0]:
            if best_val == -1 or avail[0] < best_val:
              best_val = avail[0]
              best_idx = idx
          if best_val == required[0]: # already found perfect fit
            break
        return best_idx
    
    plan = {} # mapping virtual to real id
    for acc_type in required_node_dict:
      required_list = required_node_dict[acc_type] # decreasing order
      required_list.sort(key=lambda x: -x[0])
      avail_list = avail_node_dict[acc_type]
      
      for required in required_list:
        best_idx = find_minimal_fit(required, avail_list)
        if best_idx == -1: # no solution
          return None
        avail_list[best_idx][0] -= required[0]
        
        plan[required[1]] = avail_list[best_idx][1]
    
    # solution found, tail process to apply allocation and mapping v_acc_mesh to physical
    def convert_to_acc_mesh(v_acc_mesh: Union[AcceleratorMesh, Accelerator], plan: Dict[int, int]):
      if not isinstance(v_acc_mesh, list):
        # leaf node
        v_acc: Accelerator = v_acc_mesh
        
        node_id = plan[v_acc.node_id]
        # set usage
        local_id = self.set_node_usage(node_id=node_id, mask=mask) # force the choice of correct local_id under mask
        acc = copy.copy(v_acc)
        acc.node_id = node_id
        acc.local_id = local_id
        return acc
      else:
        acc_mesh = []
        for sub_mesh in v_acc_mesh:
          acc_mesh.append(convert_to_acc_mesh(sub_mesh, plan))
        return acc_mesh
      
    p_acc_mesh = convert_to_acc_mesh(v_acc_mesh, plan)
    
    if virtual_test:
      self.free(p_acc_mesh)
        
    return p_acc_mesh
  
  """
  @EXPORT
  Allocate/Deallocate available resources given the physical acc_mesh
  
  If strict is True, the allocation will fail if any of the accelerators are already allocated. (or the inverse for deallocate)
  
  """
  def raw_allocate(self, 
                   acc_mesh: Union[AcceleratorMesh, List[Accelerator], Accelerator, None], 
                   exclude: Union[AcceleratorMesh, List[Accelerator], Accelerator, None] = None, 
                   flag: bool = True, 
                   strict: bool = True) -> bool:
    if not isinstance(acc_mesh, Iterable):
      acc_mesh = [acc_mesh] if acc_mesh is not None else []
    if not isinstance(exclude, Iterable):
      exclude = [exclude] if exclude is not None else []
    
    acc_list: List[Accelerator] = flatten_mesh(acc_mesh)
    exclude_acc_list = [(acc.node_id, acc.local_id) for acc in flatten_mesh(exclude)]
    
    acc_list = [ acc for acc in acc_list if (acc.node_id, acc.local_id) not in exclude_acc_list]
    
    for acc in acc_list:
      assert isinstance(acc, Accelerator), f"element not accelerator, got {type(acc)}"
      node_id = acc.node_id
      local_id = acc.local_id
      # set usage
      if self.get_node(node_id).usage[local_id] != flag:
        self.set_node_usage(node_id=node_id, local_id=local_id, flag=flag)
      elif strict:
        # raise RuntimeError(f"raw_allocate: accelerator {acc} not in expected {not flag} state when attempting to set to {flag}")
        return False
    return True
  
  """
  Free version of raw_allocate
  """
  def raw_free(
    self,
    acc_mesh: Union[AcceleratorMesh, List[Accelerator], Accelerator, None], 
    exclude: Union[AcceleratorMesh, List[Accelerator], Accelerator, None] = None,
    strict: bool = True
  ) -> bool:
    return self.raw_allocate(acc_mesh, exclude=exclude, flag=False, strict=strict)
  
  """
  @EXPORT
  Free resources of physical acc_mesh
  """
  def free(self, acc_mesh: Union[AcceleratorMesh, List[Accelerator], Accelerator]):
    if acc_mesh is None:
      return
    
    if not isinstance(acc_mesh, Iterable):
      acc_mesh = [acc_mesh]

    for acc in flatten_mesh(acc_mesh):
      assert isinstance(acc, Accelerator), f"element not accelerator, got {type(acc)}"
      node_id = acc.node_id
      self.reset_node_usage(node_id=node_id, local_id=acc.local_id)
      
  """
  @EXPORT
  Get cluster utilization
  """
  def get_utilization(self) -> float:
    total_acc = self.get_total_acc_num()
    used_acc = sum([node.get_usage() for node in self.cluster_node_table.values()])
    
    return used_acc / total_acc if total_acc > 0 else 0.0
      
  def _get_memory(self, acc: Union[str, Accelerator, AcceleratorMeta]) -> int:
    """
    Get the memory of the given accelerator
    """
    return self.get_acc_meta(acc).memory
    
  def _getTFLOPS(self, acc: Union[str, Accelerator, AcceleratorMeta]) -> float:
    """
    Get the TFLOPS of the given accelerator
    """
    return self.get_acc_meta(acc).TFLOPS
  
  def _sumTFLOPS(self, 
                 acc_mesh: Union[Accelerator, List]) -> float:
    """
    Sum the TFLOPS of the given accelerator mesh
    """
    if not isinstance(acc_mesh, list):
        acc_mesh = [acc_mesh]
    return sum([self.get_acc_meta(acc).TFLOPS for acc in flatten_mesh(acc_mesh)])
    
"""
  Convert acc_mesh to hostname_mesh given rm, utils for Scheduler launch
"""
def convert_to_hostname_mesh(acc_mesh: AcceleratorMesh, rm: ResourceManager) -> HostnameMesh:

  def recursive_convert(mesh):
    if not isinstance(mesh, List):
      hostname = rm.get_node(mesh.node_id).hostname
      assert hostname is not None, "hostname is None"
      return hostname
    
    result = []
    for sub_mesh in mesh:
      sub_result= recursive_convert(sub_mesh)
      result.append(sub_result)
      
    return result
  
  result = recursive_convert(acc_mesh)
  
  return result

"""
  Parser Cluster Config to ResourceManager
  :param cluster_config: Dict
  :return: ResourceManager, hostname_list
"""

def parse_cluster_config(cluster_config: Union[Dict, str]) -> Tuple[ResourceManager, List[str], Dict]:
  """
  Parse the cluster config file and return a ResourceManager object.
  """
  if isinstance(cluster_config, str):
    import json
    with open(cluster_config, 'r') as f:
      cluster_config = json.load(f)
  
  rm = ResourceManager()
  
  acc_dict = cluster_config.get("accelerator", None)
  node_dict = cluster_config.get("node", None)
  assert acc_dict is not None, "accelerator information is required"
  assert node_dict is not None, "node information is required"
  
  for type_name, acc_meta in acc_dict.items():
    acc_meta["type_name"] = type_name
    rm.register_acc_meta(**acc_meta)
  
  hostname_list = []
  for hostname, node_meta in node_dict.items():
    node_meta["hostname"] = hostname
    hostname_list.append(hostname)
    rm.register_node(**node_meta)
  
  for acc_type_or_types, bandwidth in cluster_config.get("network", {}).items():
    rm.register_bw(acc_type_or_types, bandwidth)
  
  return rm, hostname_list, cluster_config

def get_default_rm() -> ResourceManager:
  """
  Get the default resource manager with default values.
  """
  rm = ResourceManager()
  
  # register default accelerators
  rm.register_acc_meta(default_acc_type, default_acc_TFLOPS, default_acc_memory)
  
  # register default nodes
  for i in range(default_n_node):
    rm.register_node(
      acc_type=default_acc_type,
      node_id=i,
      acc_num=default_n_acc_per_node,
      hostname=f"node-{i}"
    )
  
  # register default bandwidth
  rm.register_bw_intra(default_acc_type, default_intra_bandwidth)
  rm.register_bw_intra_sat(default_acc_type, default_intra_bandwidth_saturate)
  rm.register_bw_inter(default_acc_type, default_acc_type, default_inter_bandwidth)
  
  return rm