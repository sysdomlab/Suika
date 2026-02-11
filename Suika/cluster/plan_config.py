import copy
import re 
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
from torch.nn import Module

from Suika.device_mesh import RawDeviceMesh3D
from Suika.utils import flatten_mesh, is_power_of_2

from Suika.cluster.resource import Accelerator, AcceleratorMesh, default_n_acc_per_node

if TYPE_CHECKING:
    from Suika.engine import PPEngine

LocalBatchSizeMesh = List[List[int]]
LayerPartition = List[int]

# convert nested list (disregard actual element it contains) to rank mesh (rank in increasing order)
def convert_to_rank_mesh(mesh: AcceleratorMesh) -> RawDeviceMesh3D:

    def recursive_convert(mesh, rank):  # -> result, ending rank + 1
        if not isinstance(mesh, List):
            return rank, rank + 1

        result = []
        for sub_mesh in mesh:
            sub_result, rank = recursive_convert(sub_mesh, rank)
            result.append(sub_result)

        return result, rank

    result, _ = recursive_convert(mesh, 0)

    return RawDeviceMesh3D(result)

"""
Convert acc mesh to rank mesh, while preserving existing ACC -> RANK mapping (a must for migration to work)
"""
def convert_to_rank_mesh_respectfully(acc_mesh: AcceleratorMesh, ref_acc_mesh: Optional[AcceleratorMesh]=None, ref_rank_mesh: Optional[RawDeviceMesh3D] = None) -> RawDeviceMesh3D:
    mesh3D = convert_to_rank_mesh(acc_mesh)
    if ref_acc_mesh is None or ref_rank_mesh is None:
        return mesh3D

    mesh3D = mesh3D.to_list()
    if not isinstance(ref_rank_mesh, RawDeviceMesh3D):
        ref_rank_mesh = RawDeviceMesh3D(ref_rank_mesh)

    ex_rank = max(ref_rank_mesh.flatten()) + 1
    acc_to_rank_mapping = {}

    for acc_mesh2D, rank_mesh2D in zip(ref_acc_mesh, ref_rank_mesh):
        for acc_mesh1D, rank_mesh1D in zip(acc_mesh2D, rank_mesh2D):
            for acc, rank in zip(acc_mesh1D, rank_mesh1D):
                acc_to_rank_mapping[acc] = rank
    
    for i, mesh2D in enumerate(mesh3D):
        for j, mesh1D in enumerate(mesh2D):
            for k, _ in enumerate(mesh1D):
                acc = acc_mesh[i][j][k]
                ref_rank = acc_to_rank_mapping.get(acc, None)
                if ref_rank is not None:
                    mesh1D[k] = ref_rank
                else:
                    mesh1D[k] = ex_rank
                    ex_rank += 1

    return RawDeviceMesh3D(mesh3D)


# check if the accelerator mesh is valid (2^k tp deg, and homogeneous)
def check_valid_acc_mesh(mesh: AcceleratorMesh) -> bool:
    for i in range(len(mesh)):
        for j in range(len(mesh[i])):
            tp_mesh = mesh[i][j]
            if not is_power_of_2(len(tp_mesh)):
                return False

            # check homogeneous
            acc_type_ref = tp_mesh[0].acc_type_name()
            for k in range(len(tp_mesh)):
                acc_type = tp_mesh[k].acc_type_name()
                if not acc_type == acc_type_ref:
                    return False
    return True





"""
PlanConfig
  An execution plan descriptor in both user view and cluster view
"""

class PlanConfig:

    acc_mesh: AcceleratorMesh
    lbsz_mesh: LocalBatchSizeMesh

    # default to [1]
    # regardless, lbsz_mesh should be 2D
    layer_partition: LayerPartition

    # default to 1
    num_micro_batch: int

    rank_mesh: Optional[RawDeviceMesh3D]
    
    max_tp_allowed: int # to handle cases where num_attention_head is not a times of 8, we may have to reduce tp degree.

    def __init__(self, acc_mesh: AcceleratorMesh, lbsz_mesh: LocalBatchSizeMesh, layer_partition: Optional[LayerPartition] = None, num_micro_batch: int = 1, max_tp_allowed: int = default_n_acc_per_node):
        self.acc_mesh = acc_mesh
        self.lbsz_mesh = lbsz_mesh
        self.layer_partition = layer_partition if layer_partition is not None else [1]
        self.num_micro_batch = num_micro_batch
        self.max_tp_allowed = max_tp_allowed

        self.rank_mesh = None  # used to send reconfig. set at reconfig later

    def validate(self, assertion: bool = False) -> bool:

        pp_size = len(self.acc_mesh)

        error_msg = None

        if not (pp_size == len(self.lbsz_mesh)):
            error_msg = f"Invalid PlanConfig: pp_size {pp_size} != lbsz_size {len(self.lbsz_mesh)}"
        if not (pp_size == len(self.layer_partition)):
            error_msg = f"Invalid PlanConfig: pp_size {pp_size} != layer_partition_size {len(self.layer_partition)}"

        for pp_idx in range(pp_size):
            dp_size = len(self.acc_mesh[pp_idx])
            if not dp_size == len(self.lbsz_mesh[pp_idx]):
                error_msg = f"Invalid PlanConfig: dp_size {dp_size} != lbsz_size {len(self.lbsz_mesh[pp_idx])}"

            for dp_idx in range(dp_size):
                tp_mesh = self.acc_mesh[pp_idx][dp_idx]
                tp_size = len(tp_mesh)
                if not is_power_of_2(tp_size):
                    error_msg = f"Invalid PlanConfig: tp_size {tp_size} not power of two"
                if tp_size > self.max_tp_allowed:
                    error_msg = f"Invalid PlanConfig: tp_size {tp_size} > max_tp_allowed {self.max_tp_allowed}"

                acc_type_ref = tp_mesh[0].acc_type_name()
                for i in range(tp_size):
                    acc_type = tp_mesh[i].acc_type_name()
                    if not acc_type == acc_type_ref:
                        error_msg = f"Invalid PlanConfig: not homogeneous inside TP {tp_mesh}"

                lbsz = self.lbsz_mesh[pp_idx][dp_idx]
                
                # why < 0.5 instead of <= 0? for fractional case, if < 0.5, after round off to integer case will likely be 0. it's not good if we allow such plan during optimization dynamic programming
                if lbsz < 0.5:
                    error_msg = f"Invalid PlanConfig: lbsz {lbsz} at f{(pp_idx, dp_idx)} < 0.5"

            num_layer = self.layer_partition[pp_idx]
            if num_layer < 0.5:
                error_msg = f"Invalid PlanConfig: num_layer {num_layer} at pp_idx {pp_idx} < 0.5"

        if assertion:
            assert error_msg is None, error_msg

        return error_msg is None

    """
    UTILS
    """
    
    """
    convert to PPEngine kwargs
    
    Args:
        model_provider (Callable[[int, int], Module]) = None: 
            It should take (layer_index, n_layer) as input and returns a model instance
            If None, assumes "model" field is already set in kwargs.
        **kwargs: Additional keyword arguments to be passed.
    
    Returns:
        Dict: A dictionary of arguments for PPEngine.
    
    Note:
        This method is only valid after distributed process group is initialized.
    """
    def to_PPEngine_kwargs(self, model_provider: Callable[[int, int], "Module"] = None, **kwargs) -> Dict:
        kwargs = dict(**kwargs)
        
        # prepare device mesh and coord
        if self.rank_mesh is None:
            # should only happen at initial launch, but not at reconfiguration, so ref_mesh is not needed
            self.set_rank_mesh()
        
        device_mesh = self.rank_mesh.get_dist_initialized()
        rank = torch.distributed.get_rank()
        pp_rank, dp_rank, tp_rank = device_mesh.get_coord(rank)
        
        if "model" not in kwargs:
            assert model_provider is not None, "model_provider must be provided if 'model' is not in kwargs"
            layer_index = sum(self.layer_partition[:pp_rank])
            n_layer = self.layer_partition[pp_rank]
            kwargs["model"] = model_provider(layer_index, n_layer)
        
        if "device" not in kwargs:
            print("Warning: device not set in to_PPEngine kwargs, default to cuda:0")
            kwargs["device"] = "cuda:0"
        
        kwargs["mesh"] = device_mesh        
        kwargs["lbsz_mesh"] = self.lbsz_mesh
        kwargs["num_micro_batch"] = self.num_micro_batch
        
        return kwargs
    
    def to_PPEngine(self, model_provider: Callable[[int, int], "Module"] = None, **kwargs) -> "PPEngine":
        from Suika.engine import PPEngine

        kwargs = self.to_PPEngine_kwargs(model_provider, **kwargs)
        return PPEngine(**kwargs)
        
        
    # must round for these functions
    # as load balance algorithm use allocated < total as termination condition for integer case,
    # without this, if we call load balance for integer case on a fractional plan, an addtional layer or batch size may be added due to eps error.
    
    def global_batch_size(self) -> int:
        return self.global_micro_batch_size() * self.num_micro_batch

    def global_micro_batch_size(self) -> int:
        return int(round(
            sum(self.lbsz_mesh[0])
        ))

    def total_num_layer(self) -> int:
        return int(round(
            sum(self.layer_partition)
        ))
    
    def acc_list(self) -> List[Accelerator]:
        return flatten_mesh(self.acc_mesh)
        
    def world_size(self) -> int:
        return len(self.acc_list())

    def instantiate(self, acc_mesh: AcceleratorMesh) -> "PlanConfig":
        instance = copy.deepcopy(self)
        instance.acc_mesh = acc_mesh
        return instance

    def set_rank_mesh(self, ref_acc_mesh: Optional[AcceleratorMesh] = None, ref_rank_mesh: Optional[RawDeviceMesh3D] = None) -> RawDeviceMesh3D:
        self.rank_mesh = convert_to_rank_mesh_respectfully(self.acc_mesh, ref_acc_mesh, ref_rank_mesh)
        return self.rank_mesh
    
        
    def __str__(self):
        result = "PlanConfig(\n"
        indent = 2

        pp_deg = len(self.acc_mesh)
        result += f"{indent * ' '}PP[{pp_deg}]:{{\n"
        result += f"{indent * ' '}  > num_micro_batch: {self.num_micro_batch}\n"

        for i in range(pp_deg):
            indent += 2

            dp_deg = len(self.acc_mesh[i])
            result += f"{indent * ' '}DP({i})[{dp_deg}]:{{\n"
            result += f"{indent * ' '}  > n_layer: {self.layer_partition[i]}\n"
            for j in range(dp_deg):
                indent += 2

                tp_deg = len(self.acc_mesh[i][j])
                result += f"{indent * ' '}TP({j})[{tp_deg}]:{{\n"
                result += f"{indent * ' '}  > lbsz: {self.lbsz_mesh[i][j]}\n"
                for k in range(tp_deg):
                    indent += 2
                    result += f"{indent * ' '}{self.acc_mesh[i][j][k]}" + (f"-({self.rank_mesh.get_rank((i, j, k))})" if self.rank_mesh else "") + "\n"
                    indent -= 2

                result += f"{indent * ' '}}}\n"
                indent -= 2

            result += f"{indent * ' '}}}\n"
            indent -= 2

        result += f"{indent * ' '}}}\n"
        result += f")\n"
        return result

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PlanConfig):
            return False

        return (
            self.acc_mesh == other.acc_mesh and
            self.lbsz_mesh == other.lbsz_mesh and
            self.layer_partition == other.layer_partition and
            self.num_micro_batch == other.num_micro_batch
        )

    """
      convert a str back to planconfig
      
    """
    @staticmethod
    def from_str(plan_config_str: str, fractional: bool = False) -> "PlanConfig":
        """
        PlanConfig(
        )
        """
        dtype_convert = float if fractional else int
        
        lines = [line.strip() for line in plan_config_str.strip().split('\n') if line.strip()]
        if not lines:
            raise ValueError("Empty PlanConfig string")

        line_iter = iter(lines)

        def parse_value_from_line(line: str, prefix: str) -> str:
            return line.split(prefix, 1)[1].strip()

        # Consume "PlanConfig("
        if not next(line_iter).startswith("PlanConfig("):
            raise ValueError("Invalid PlanConfig string: Missing 'PlanConfig('")

        acc_mesh: AcceleratorMesh = []
        lbsz_mesh: LocalBatchSizeMesh = []
        layer_partition: LayerPartition = []
        
        rank_mesh: List[List[List[int]]] = []


        # Outer loop for PP stages
        current_line = next(line_iter, "")

        if not current_line.startswith("PP["): # PP
            raise ValueError(f"Expected PP stage, got '{current_line}'")

        line_after_pp_header = next(line_iter)
        if not line_after_pp_header.startswith("> num_micro_batch:"):
            raise ValueError(f"Expected '> num_micro_batch:', got '{line_after_pp_header}'")
          
        num_micro_batch = int(parse_value_from_line(line_after_pp_header, "> num_micro_batch:"))

        current_line = next(line_iter, "")
        
        while current_line.startswith("DP("): # DP
            stage_acc_mesh: List[List[Accelerator]] = []  # acc_mesh[pp_idx]
            stage_lbsz: List[int] = []  # lbsz_mesh[pp_idx]
            stage_rank_mesh: List[List[int]] = []  # rank_mesh[pp_idx]
        
            n_layer_line = next(line_iter)
            if not n_layer_line.startswith("> n_layer:"):
                raise ValueError(f"Expected '> n_layer:', got '{n_layer_line}'")
            layer_partition.append(dtype_convert(parse_value_from_line(n_layer_line, "> n_layer:")))

            current_line = next(line_iter)
            while current_line.startswith("TP("): # TP
                tp_acc_mesh: List[Accelerator] = []
                tp_rank_mesh: List[int] = []

                lbsz_line = next(line_iter)
                if not lbsz_line.startswith("> lbsz:"):
                    raise ValueError(f"Expected '> lbsz:', got '{lbsz_line}'")
                lbsz_val = dtype_convert(parse_value_from_line(lbsz_line, "> lbsz:"))
                stage_lbsz.append(lbsz_val)

                while True: # accelerator
                    current_line = next(line_iter)
                    if current_line.startswith("Accelerator("):
                        acc, rank = PlanConfig._accelerator_from_str(current_line)
                        tp_acc_mesh.append(acc)
                        tp_rank_mesh.append(rank)
                    elif current_line == "}":  # End of current TP mesh
                        break
                    else:
                        raise ValueError(f"Expected Accelerator or '}}' for TP, got '{current_line}'")

                stage_acc_mesh.append(tp_acc_mesh)
                stage_rank_mesh.append(tp_rank_mesh)
                current_line = next(line_iter)

            if current_line != "}":
                raise ValueError(f"Expected '}}' to close DP block, got '{current_line}'")

            current_line = next(line_iter)

            acc_mesh.append(stage_acc_mesh)
            lbsz_mesh.append(stage_lbsz)
            rank_mesh.append(stage_rank_mesh)

        if current_line != "}":
            raise ValueError(f"Expected '}}' to close PP block, got '{current_line}'")

        plan_config = PlanConfig(acc_mesh, lbsz_mesh, layer_partition, num_micro_batch)
        
        if all(isinstance(r, int) for r in flatten_mesh(rank_mesh)):
          plan_config.rank_mesh = RawDeviceMesh3D(rank_mesh)
        else:
          plan_config.rank_mesh = None

        return plan_config
      
    @staticmethod
    def _accelerator_from_str(acc_str: str) -> Tuple[Accelerator, Optional[int]]:
        # Example: "Accelerator(acc_type=H100, id=(0, 0))"
        # Or "            Accelerator(acc_type=H100, id=(0, 0))-(12)"

        rank: Optional[int] = None
        parts = acc_str.strip().split('-(')
        acc_str_cleaned = parts[0].strip()

        if len(parts) > 1:
            rank_str = parts[1]
            if rank_str.endswith(')'):
                try:
                    rank = int(rank_str[:-1])
                except ValueError:
                    raise ValueError(f"Invalid rank format in Accelerator string: {acc_str}")
            else:
                raise ValueError(f"Invalid rank format in Accelerator string: {acc_str}")

        match = re.match(r"Accelerator\(acc_type=([^,]+),\s*id=\((\d+),\s*(\d+)\)\)", acc_str_cleaned)
        if not match:
            raise ValueError(f"Invalid Accelerator string format: {acc_str_cleaned}")

        acc_type = match.group(1)
        node_id = int(match.group(2))
        local_id = int(match.group(3))

        return Accelerator(acc_type=acc_type, node_id=node_id, local_id=local_id), rank


"""
    parse_plan_config_from_str
    Args: 
        plan_config_str (str): The string representation of the PlanConfig.
        fractional (bool): Whether to parse local batch sizes as floats or ints.
    Returns:
        PlanConfig: The parsed PlanConfig object.
"""
def parse_plan_config_from_str(plan_config_str: str, fractional: bool = False) -> PlanConfig:
    return PlanConfig.from_str(plan_config_str, fractional=fractional)

"""
Parse a PlanConfig from a file
"""
def parse_plan_config_from_file(file: str, fractional: bool = False):
    with open(file, "r") as f:
        s = "\n".join(f.readlines())

    return parse_plan_config_from_str(s, fractional=fractional)

"""
    get_uniform_plan_config
    Args:
        accelerator (str): The type of accelerator.
        pp_dp_tp (Tuple[int, int, int]): The PP, DP, and TP degree.
        gbs (int): The global batch size.
        n_layer (int): The number of layers.
        n_mb (int): The number of micro-batches.
        is_micro_gbs (bool): Whether to GBS is micro GBS.
        fractional (bool): Whether to use fractional local batch sizes.
        compact_allocation (bool): Whether to force plan to be allocated as compact as possible (i.e, takes the minimum number of nodes as Sia describes)
        n_acc_per_node (int): Only effective if compact_allocation is True
        
    Returns:
        PlanConfig: The generated PlanConfig object.
        
    Note:
        when fractional is False, micro_gbs / dp, n_layer / pp will be rounded down, remainder added to firsts.
        
        gbs / n_mb is rounded down strictly, as no support for dynamic lbsz across micro batches.
"""

def get_uniform_plan_config(
    acc_type: str, 
    pp_dp_tp: Tuple[int, int, int], 
    gbs: int, 
    n_layer: int, 
    n_mb: int, 
    is_micro_gbs = False, 
    fractional: bool = False,
    compact_allocation: bool = True,
    n_acc_per_node: int = default_n_acc_per_node,
    max_tp_allowed: int = default_n_acc_per_node
    ) -> PlanConfig:

    micro_gbs = gbs if is_micro_gbs else gbs // n_mb

    if not is_micro_gbs and gbs % n_mb != 0:
        print(f"WARNING: gbs {gbs} is not divisible by n_mb {n_mb}")

    pp_deg, dp_deg, tp_deg = pp_dp_tp
    
    def div(num, den, idx):
        if fractional:
            return num / den
        else:
            return num // den + (idx < num % den)
    
    def node_id(pp_id, dp_id, tp_id):
        if compact_allocation:
            rank = (pp_id * (dp_deg * tp_deg) + dp_id * tp_deg + tp_id)
            return rank // n_acc_per_node
        else:
            return pp_id * dp_deg + dp_id
        
    def local_id(pp_id, dp_id, tp_id):
        if compact_allocation:
            rank = (pp_id * (dp_deg * tp_deg) + dp_id * tp_deg + tp_id)
            return rank % n_acc_per_node
        else:
            return tp_id


    acc_mesh = [
        [
            [
                Accelerator(acc_type=acc_type, node_id=node_id(pp_id, dp_id, tp_id), local_id=local_id(pp_id, dp_id, tp_id))
                for tp_id in range(tp_deg)
            ] for dp_id in range(dp_deg)
        ] for pp_id in range(pp_deg)
    ]
    
    lbsz_mesh = [
            [div(micro_gbs, dp_deg, dp_id) for dp_id in range(dp_deg)] 
    ] * pp_deg
    
    layer_partition = [
        div(n_layer, pp_deg, pp_id) for pp_id in range(pp_deg)
    ]

    return PlanConfig(
        acc_mesh=acc_mesh,
        lbsz_mesh=lbsz_mesh,
        layer_partition=layer_partition,
        num_micro_batch=n_mb,
        max_tp_allowed=max_tp_allowed
    )