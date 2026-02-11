"""
Python implementation of the original C++ Engine, including estimator and solver
"""
from Suika.cluster.resource import Accelerator, ResourceManager, default_n_acc_per_node
from Suika.cluster.plan_config import LayerPartition, LocalBatchSizeMesh, PlanConfig, check_valid_acc_mesh
from Suika.utils import flatten_mesh

import copy
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING, Union
import heapq
import logging
from logging import Logger
import math
import os
import time

"""
    Backend Comments:
    
    Due to simulation performance concerns, we reimplement the optimization algorithm in C++
    
    Python backend (implemented in this file) reflects the same algorithm as the C++ backend (under /Suika/cluster/cpp_backend), and is operational.
    
    The result of two backends may differ slightly due to floating point precision and nuanced details of implementation, but the algorithm is completely the same.
    
    To switch backend:
        Set the environment variable SUIKA_BACKEND to "CPP" or "PYTHON".
        
    Default backend:
        When cpp-compiled module is present, the C++ backend is default. Otherwise, fall back to python.
        
    To compile C++ backend:
        Go to cpp_backend folder, python setup.py build_ext --inplace
"""

back_end_type = os.environ.get("SUIKA_BACKEND", "CPP").upper()
assert back_end_type in ["CPP", "C++", "PYTHON", "PY"], f"SUIKA_BACKEND must be one of 'CPP', 'C++' or 'PYTHON', 'PY', got {back_end_type}"

if back_end_type in ["CPP", "C++"]:
    try:
        from Suika.cluster.cpp_backend import optimize_all as optimize_all_cpp
    except ImportError:
        print("Warning: Suika C++ backend not found; fall back to Python backend's optimize_all. Did you forget to install?")
        optimize_all_cpp = None
else:
    optimize_all_cpp = None

AcceleratorMesh3D = List[List[List[Accelerator]]]
AcceleratorMesh2D = List[List[Accelerator]]
AcceleratorMesh1D = List[Accelerator]

OVERRIDE_CHECK_VALID_MESH = False
ESTIMATOR_NO_LOGGING = True
FEASIBILITY_MARGIN = 0.10 # reserve (1 + FEASIBILITY_MARGIN) GPU memory for safety
EPS = 1e-3 # add EPS at _lb_compare_max for maximum C++/Python backend consistency
ENABLE_SELF_ADJUSTMENT = True # whether to enable self adjustment in optimize_all
ENABLE_LIMITED_EXPAND = True
ENABLE_FRACTIONAL_LB = False
ENABLE_LIMITED_PREDECESOR = True
MAX_PREDECESSOR = 4

ABSOLUTE_IMPROVEMENT_MARGIN = 0.1 # minimum absolute throughput improvement margin to be accepted as reconfig

class Estimator():
  
    C: Dict[str, float] # constants
    
    rm: ResourceManager
    
    """
        Constants fields are :
    """
    constant_fields: List[str] = [
        "comp", # TFLOPS per layer per sample (not partitioned by tp) (including forward only)
        "optimizer", # TFLOPS per layer (not partitioned by tp) for optimizer latency
        "param", # size of (trainable) param in MB per layer (not partitioned by tp) for DP communication
        "activation", # size of single activation in MB per layer per sample (not partitioned by tp) for TP communication
        "backward", # ratio of backward time to forward time in computation
        "comp_commu_overlap", # exponential parameter of communication overlap with computation for DP
        "param_optim_mem", # size of (all) param in MB per layer (not partitioned by tp) for memory footprint, including grad and optimizer state
        "activation_mem_np", # size of activation in MB per layer per sample not partitionable by tp for memory footprint
        "activation_mem_p", # size of activation in MB per layer per sample partitionable by tp for memory footprint
    ]
    
    block_update: bool = False
    
    logging_interval = 300
    
    def __init__(self, rm: ResourceManager, C: Dict[str, float] = None, logger: Optional[Logger] = None):
        self.C = {}
        if C is not None:
            self.C.update(C) # avoid referencing original C (otherwise simulator will fail)

        self.rm = rm
        
        self.block_update = False

        self.logger = logger
        self.last_logging_time = 0

    def set_constant(self, name: str, value: float):
        """
        Set the constant value for the given name.
        """
        self.C[name] = value

    def set_constant_dict(self, C: Dict[str, float]):
        """
        Set the constant dictionary.
        """
        self.C.update(C)
        
    def update_constants_from_profile(self, C_raw: Dict[str, float], plan_config: PlanConfig):
        """
        Calucate and set the constant values.
        C_raw:
            see engine.py:PPEngine.profile_result_to_scheduler
        """
        if plan_config is None or self.block_update:
            return
        
        # calculate the constants
        C = {}
        num_layer = plan_config.layer_partition[0]
        lbsz = plan_config.lbsz_mesh[0][0]
        tp_deg = len(plan_config.acc_mesh[0][0])

        ## param / activation
        C["param"] = C_raw["param_raw"] / num_layer * tp_deg
        C["activation"] = C_raw["activation_raw"] / lbsz

        C["param_optim_mem"] = C_raw["param_optim_mem_raw"] / num_layer * tp_deg

        activation_mem_np_scale = C_raw.get("activation_mem_np_scale", 5.0)
        C["activation_mem_np"] = activation_mem_np_scale * C["activation"]
        C["activation_mem_p"] = (C_raw["activation_mem_raw"] / num_layer / lbsz - C["activation_mem_np"]) * tp_deg

        ## comp
        mesh = plan_config.acc_mesh[0][0]
        min_TFLOPS = min(map(lambda acc: self.rm._getTFLOPS(acc), mesh))
        
        if tp_deg == 1:
            TP_commu_fwd = 0    
        else:
            volume = 2 * C["activation"] * 2 * (tp_deg - 1) / tp_deg # forward only takes 1/2 communication
            bandwidth = min(self.rm.get_weak_bandwidth(mesh[i], mesh[j], volume / 2) for i in range(tp_deg) for j in range(i + 1, tp_deg))
            
            TP_commu_fwd = volume / bandwidth
            TP_commu_fwd *= lbsz * num_layer
        # self.log(f"estimated forward commu vs raw forward commu: {TP_commu_fwd} / {C_raw['forward_commu_raw']}")
        
        C["comp"] = (C_raw["forward_comp_raw"] / num_layer / lbsz) * min_TFLOPS * tp_deg
        C["optimizer"] = (C_raw["optimizer_raw"] / num_layer) * min_TFLOPS * tp_deg
        
        ## backward
        C["backward"] = C_raw["backward"] # ratio of backward time to forward time
        
        ## comp_commu_overlap
        emprical_overlap = 1.0
        eps = 1e-3
        if C_raw["backward_dp_commu_raw"] <= eps:
            C["comp_commu_overlap"] = emprical_overlap
        else:
            self.set_constant_dict(C) # we need call _dp_latency with param field set.
            dp_comp_bwd, dp_commu = self._dp_latency(mesh=plan_config.acc_mesh[0], 
                                                   lbszs=plan_config.lbsz_mesh[0], 
                                                   num_layer=num_layer,
                                                   skip_commu_olp=True
                                                   )
            
            if dp_commu != 0:
                C["comp_commu_overlap"] = self._fit_comp_commu_overlap(dp_comp_bwd, dp_commu, C_raw["backward_dp_commu_raw"] )
            else:
                C["comp_commu_overlap"] = emprical_overlap
        self.set_constant_dict(C)
        # self.log(f"update constants: {C}")
        
        # check if estimated latency matches the raw latency
        if "iter_time" in C_raw:
            est_latency = self.estimate(plan_config, test_feasbile=True)
            raw_latency = C_raw["iter_time"]
            relative_error = abs(est_latency - raw_latency) / raw_latency
            # self.log(f"estimated latency vs raw latency: {est_latency} / {raw_latency}; error={relative_error*100:.2f}%")
        else:
            relative_error = 0.0
        
        # logging
        now = time.time()
        if now - self.last_logging_time > self.logging_interval:
            self.last_logging_time = now
            if relative_error * 100 > 10:
                self.log(f"estimated latency vs raw latency: {est_latency} / {raw_latency}; error={relative_error*100:.2f}%", level=logging.DEBUG, override_no_logging=True)
                self.log(f"relative diff is too large: {relative_error * 100:.2f}%", level=logging.DEBUG, override_no_logging=True)    
            if C["optimizer"] == 0.0:
                self.log(f"optimizer not profiled", level=logging.DEBUG, override_no_logging=True)

            self.log(f"estimator_C: {self.C}", level=logging.DEBUG, override_no_logging=True)
        
        return
    
    def set_block_update_constants(self, flag: bool):
        """
        Block constant update during reconfiguration
        """
        if flag:
            self.C = {}
        self.block_update = flag
    
        
    def initialized(self):
        return all([field in self.C for field in self.constant_fields])

    def estimate(self, plan_config: PlanConfig, test_feasbile = False, log_prefix: Optional[str] = None, output_accum_and_optim_step_time: bool = False, comp_decay: float = None) -> float:
        """
        Estimate the latency of the given plan config (no feasibility test)
        # comp decay: 
            decay the T_comp by ratio of comp_decay to simulate underutilization. only used with simulator estimation.
        """    
    
        if not self.initialized() or plan_config is None or (test_feasbile and not self.feasible(plan_config, log_prefix=log_prefix, detailed_logging=True)):
            return float("inf")
            
    
        latency = self._pp_latency(
            mesh=plan_config.acc_mesh,
            lbsz_mesh=plan_config.lbsz_mesh, 
            layer_partition=plan_config.layer_partition,
            num_micro_batch=plan_config.num_micro_batch,
            log_prefix=f"{log_prefix}" if log_prefix is not None else None,
            output_accum_and_optim_step_time=output_accum_and_optim_step_time,
            comp_decay=comp_decay,
            )
        
        if log_prefix is not None:
            self.log(f"{log_prefix}: latency: {latency}")
            
        return latency
    
    def feasible(self, plan_config: PlanConfig, log_prefix: Optional[str] = None, detailed_logging: bool = False, output_peak_usage: bool = False) -> Union[bool, float]:
        """
        Check if the given plan config is feasible (OOM)
        """
        if plan_config is None:
            return False if not output_peak_usage else float("inf")
        if not plan_config.validate():
            return False if not output_peak_usage else float("inf")
        if not self.initialized():
            return False if not output_peak_usage else float("inf")
        
        peak_usage = 0
        peak_usage_at = None
        
        pp_deg = len(plan_config.acc_mesh)
        num_micro_batch = plan_config.num_micro_batch
        
        for i in range(pp_deg):
            dp_deg = len(plan_config.acc_mesh[i])
            num_layer = plan_config.layer_partition[i]
            for j in range(dp_deg):
                tp_deg = len(plan_config.acc_mesh[i][j])
                lbsz = plan_config.lbsz_mesh[i][j]
                for k in range(tp_deg):
                    acc: Accelerator = plan_config.acc_mesh[i][j][k]
                    available = self.rm._get_memory(acc)

                    activation_per_layer = lbsz * min(pp_deg - i, num_micro_batch) * (self.C["activation_mem_p"] / tp_deg + self.C["activation_mem_np"])
                    param_per_layer = self.C["param_optim_mem"] / tp_deg
                    grad_per_layer = self.C["param"] / tp_deg if \
                                    num_micro_batch <= pp_deg - i \
                                    else 0 # whether the first backward comes before or after all forward.
                    required = num_layer * (activation_per_layer + param_per_layer - grad_per_layer)
                    required = required * (1 + FEASIBILITY_MARGIN)

                    if available < required:
                        if log_prefix is not None:
                            self.log(f"{log_prefix}: infeasible@{(i, j, k)}, "\
                                     f"available={available}, "\
                                     f"required={required} with {num_layer} layers, {lbsz} lbsz, {tp_deg} tp_deg")
                        if not output_peak_usage:
                            return False
                    if required / available > peak_usage:
                        peak_usage = required / available
                        peak_usage_at = (i, j, k)

                    if detailed_logging and log_prefix is not None:
                        self.log(f"{log_prefix}: @{(i, j, k)}, "\
                                 f"available={available}, "\
                                 f"required={required} with {num_layer} layers, {lbsz} lbsz, {tp_deg} tp_deg")
        
        if log_prefix is not None:
            self.log(f"{log_prefix}: feasible")
            self.log(f"{log_prefix}: peak usage@{peak_usage_at} {peak_usage * 100:.2f}%")
        if not output_peak_usage:
            return True
        else:
            return peak_usage
        
        
    def log(self, msg: str, level=logging.DEBUG, override_no_logging: bool = False):
        if ESTIMATOR_NO_LOGGING and level == logging.DEBUG and not override_no_logging:
            return
        
        if self.logger is not None:
            pattern_msg = f"Estimator-{msg}"
            self.logger.log(level, pattern_msg)
        
    def _tp_latency(self, mesh: AcceleratorMesh1D, lbsz: int, num_layer: int, log_prefix: Optional[str] = None, comp_decay: float=None):
        """
        Estimate the compute/communication latency of tensor parallelism for 1 micro-batch
        """

        
        tp_deg = len(mesh)

        # comp
        min_TFLOPS = min(map(lambda acc: self.rm._getTFLOPS(acc), mesh))
        TP_comp_fwd = 1 * self.C["comp"] / min_TFLOPS / tp_deg
        TP_comp_bwd = self.C["backward"] * TP_comp_fwd
        
        # commu
        if tp_deg == 1:
            TP_commu = 0
        else:
            # only forward / backward, seperated, so the factor is 2 instead of 4           
            volume = 2 * self.C["activation"] * 2 * (tp_deg - 1) / tp_deg
            bandwidth = min(self.rm.get_weak_bandwidth(mesh[i], mesh[j], volume / 2) for i in range(tp_deg) for j in range(i + 1, tp_deg))
            TP_commu = volume / bandwidth

        scaling = lbsz * num_layer
        if comp_decay is not None:
            scaling *= max(comp_decay, 1 / comp_decay)
        TP_comp_fwd *= scaling
        TP_comp_bwd *= scaling
        TP_commu *= scaling

        TP_fwd = TP_comp_fwd + TP_commu
        TP_bwd = TP_comp_bwd + TP_commu
        
        # optim
        TP_optim = self.C["optimizer"] / min_TFLOPS / tp_deg * num_layer

        if log_prefix is not None:
            self.log(f"{log_prefix}: TP_comp_fwd: {TP_comp_fwd:.4f}, TP_comp_bwd: {TP_comp_bwd:.4f}, TP_commu: {TP_commu:.4f}")
            self.log(f"{log_prefix}: TP_fwd: {TP_fwd:.4f}, TP_bwd: {TP_bwd:.4f}, TP_optim: {TP_optim:.4f}")
            self.log(f"{log_prefix}: tp_deg={tp_deg}, lbsz={lbsz}, num_layer={num_layer}")
        return TP_fwd, TP_bwd, TP_optim
      
    def _dp_latency(self, mesh: AcceleratorMesh2D, lbszs: List[int], num_layer: int, log_prefix: Optional[str] = None, skip_commu_olp: bool = False, comp_decay: float=None):
        """
        Estimate the compute latency of data parallelism for 1 micro-batch
                     communication latency for 1 mini-batch
        """
        dp_deg = len(mesh)
        
        # comp
        TP_fwd_bwd_optim = [
                self._tp_latency(
                    mesh=mesh[i], 
                    lbsz=lbszs[i],
                    num_layer=num_layer,
                    log_prefix=f"{log_prefix}({i})TP" if log_prefix is not None else None,
                    comp_decay=comp_decay
                ) 
                for i in range(dp_deg)
            ]
        TP_fwd, TP_bwd, TP_optim = map(list, zip(*TP_fwd_bwd_optim))
        DP_comp_fwd = max(TP_fwd)
        DP_comp_bwd = max(TP_bwd)
        DP_optim = max(TP_optim)

        # commu
        if dp_deg == 1:
            DP_commu = 0
        else:
            total_volume = num_layer * self.C["param"] * 2 * (dp_deg - 1) / dp_deg # volume of param per tp group 
            
            # DP_commu = volume / bandwidth
            def get_dp_commu_latency(tp_mesh1: AcceleratorMesh1D, tp_mesh2: AcceleratorMesh1D) -> float:
                tp_deg1 = len(tp_mesh1)
                tp_deg2 = len(tp_mesh2)
                if tp_deg1 > tp_deg2:
                    tp_mesh1, tp_mesh2 = tp_mesh2, tp_mesh1
                    tp_deg1, tp_deg2 = tp_deg2, tp_deg1
                
                # assume tp_deg1 <= tp_deg2
                scale = tp_deg2 // tp_deg1
                bandwidth = min([
                    self.rm.get_bandwidth(tp_mesh1[i], tp_mesh2[i * scale + j])    
                    for i in range(tp_deg1) for j in range(scale)
                ])
                return total_volume / tp_deg1 / bandwidth
            
            DP_commu = max(
                [
                    get_dp_commu_latency(mesh[i], mesh[j])
                    for i in range(dp_deg) for j in range(i + 1, dp_deg)
                ]
            )
        
        if skip_commu_olp:
            return DP_comp_bwd, DP_commu # used for fitting comp_commu_overlap

        DP_commu_no_olp = self._commu_overlapped_latency(T_comp_bwd=DP_comp_bwd, T_commu=DP_commu)
        DP_commu_optim_no_olp = DP_commu_no_olp + DP_optim
            
        if log_prefix is not None:
            self.log(f"{log_prefix}: DP_comp_fwd: {DP_comp_fwd:.4f}, DP_comp_bwd: {DP_comp_bwd:.4f}, DP_optim: {DP_optim:.4f}")
            self.log(f"{log_prefix}: DP_commu: {DP_commu:.4f}, DP_commu_optim_no_olp: {DP_commu_optim_no_olp:.4f}")
            self.log(f"{log_prefix}: dp_deg={dp_deg}")
            
        return (DP_comp_fwd, DP_comp_bwd, DP_commu_optim_no_olp)

    def _pp_latency(self, mesh: AcceleratorMesh3D, lbsz_mesh: LocalBatchSizeMesh, layer_partition: LayerPartition, num_micro_batch: int, log_prefix: Optional[str] = None, output_accum_and_optim_step_time: bool = False, comp_decay: float=None):
        """
        Estimate the latency of pipeline parallelism for entire 
        """
        if log_prefix is not None:
            log_prefix = f"{log_prefix}PP"
        
        pp_deg = len(mesh)
        
        dp_latencys = [
                self._dp_latency(
                mesh=mesh[i], 
                lbszs=lbsz_mesh[i], 
                num_layer = layer_partition[i],
                log_prefix=f"{log_prefix}({i})DP" if log_prefix is not None else None,
                comp_decay=comp_decay) for i in range(pp_deg)
        ]
        
        DP_comp_fwds, DP_comp_bwds, DP_commu_optim_no_olps = map(list, zip(*dp_latencys))
        DP_comps = list(map(sum, zip(DP_comp_fwds, DP_comp_bwds)))

        sum_DP_comp = sum(DP_comps)
        max_DP_comp = max(DP_comps)
        max_DP_commu_optim_no_olp = max([
               DP_commu_optim_no_olps[i] - sum(DP_comps[:i]) for i in range(pp_deg)
        ])

        latency = sum_DP_comp + (num_micro_batch - 1) * max_DP_comp + max_DP_commu_optim_no_olp
        
        if log_prefix is not None:
            self.log(f"{log_prefix}: sum_DP_comp: {sum_DP_comp:.4f}, max_DP_comp: {max_DP_comp:.4f}")
            self.log(f"{log_prefix}: max_DP_commu_optim_no_olp: {max_DP_commu_optim_no_olp:.4f}")
            self.log(f"{log_prefix}: latency: {latency:.4f}")
            self.log(f"{log_prefix}: num_micro_batch={num_micro_batch}")
            
        if not output_accum_and_optim_step_time:
            return latency
        else:
            return sum_DP_comp, sum_DP_comp + max_DP_commu_optim_no_olp

    def _commu_overlapped_latency(self, T_comp_bwd, T_commu, k_overlap = None):
        k = self.C["comp_commu_overlap"] if k_overlap is None else k_overlap
        
        T_comp_bwd = max(T_comp_bwd, 0.0) # avoid math error
        T_commu = max(T_commu, 0.0)
        return math.pow(math.pow(T_comp_bwd, k) + math.pow(T_commu, k), 1/k) - T_comp_bwd
    
    def _fit_comp_commu_overlap(self, T_comp_bwd: float, T_commu: float, T_commu_raw: float) -> float:
        low, high = 1.0, 10.0
        while low < high - 1e-3:
            mid = (low + high) / 2
            evaluated = self._commu_overlapped_latency(T_comp_bwd, T_commu, mid)
            if evaluated < T_commu_raw:
                high = mid
            else:
                low = mid
        return low
        
class JobOptimizer():
    rm: ResourceManager
    est: Estimator
    
    def __init__(self, rm: ResourceManager, C = None, logger: Optional[Logger] = None):
        self.rm = rm
        self.est = Estimator(rm=rm, C=C, logger=logger)
        self.logger = logger
        
    def eval_latency(self, plan_config: PlanConfig, log_prefix=None, **kwargs) -> float:
        """
        Evaluate the latency of the given plan config
        """
        
        # assert plan_config is not None, "plan_config is None"        
        return self.est.estimate(plan_config, test_feasbile=True, log_prefix=log_prefix, **kwargs)
        
    def eval_throughput(self, plan_config: PlanConfig, log_prefix=None, **kwargs) -> float:
        """
        Evaluate the throughput of the given plan config
        """
        latency = self.eval_latency(plan_config, log_prefix=log_prefix, **kwargs)
        if latency < 0 or latency == float("inf"): # infeasible
            return 0
        else:
            return 1 / latency
    
    def eval_speed_up(self, 
                      plan_config: PlanConfig, 
                      new_resources: AcceleratorMesh2D, 
                      ) -> Tuple[PlanConfig, float]:
        """
        Evaluate the speed up of the given plan config
        
        deprecated, use eval_effective_speed_up instead
        """
        if not self.est.initialized():
            return (None, -1)
        
        new_plan_config = self.optimize(cur_plan_config=plan_config, 
                                        new_acc_units=new_resources,
                                        )
        new_throughput = self.eval_throughput(new_plan_config)
        old_throughput = self.eval_throughput(plan_config)
        
        speed_up = 0
        if new_throughput == 0: # new plan infeasible
            speed_up = -1
        elif old_throughput == 0: # infeasible -> feasible
            speed_up = float("inf")
        else:
            speed_up = new_throughput / old_throughput - 1 
        return new_plan_config, speed_up
    
    def eval_best_effective_speed_up(
                        self,
                        cur_plan_config: PlanConfig,
                        new_resources: AcceleratorMesh2D,
                        rescaling: Optional[Callable[[AcceleratorMesh1D], float]] = None,
                        optimize_all_fn: Callable[[PlanConfig, AcceleratorMesh2D], Tuple[List[Tuple[PlanConfig, float]], AcceleratorMesh2D]] = None,
    ) -> Tuple[PlanConfig, float]:
        """
        Evaluate the effective speed up of the given plan config and new resources
        
        Main interface for cluster_wide scheduler
        
        Args:
            cur_plan_config: the current plan config
            new_resources: the new resources to be added
            rescaling: a function to rescale the effective speed up
            optimize_all_fn: a drop-in function to replace optimize_all for baseline (DP/Metis), if not provided, use self.optimize_all (Suika)

        Returns:
            new_plan_config: the new plan config
            effective_speed_up: the effective speed up correspond to new plan config
        

        Roughly speaking, 
            effective_speed_up measures how efficiently a task utilizes new accelerators compared to
            ideal linear TFLOPS scaling. It answers: "How close are we to perfect scaling?"
            
            ESU = 0 means no improvement over the baseline (old throughput)
            ESU = 1 means perfect linear throughput scaling
    
            a rescaling function may be provided to achieve cluster-wide scheduling goal.
            (e.g., scale ESU down with more accelerators to prefer finer-grained reconfiguration) 
        
        Formally,
            ESU = 
               rescaling(R) * (new_throughput(R) - old_throughput) / (ideal_linear_speed_up(R) - old_throughput)
            for R as a subset of new_resources
                        
        The function find the near-optimal plan using (a subset of) new_resources guided by the ESU
        """
        if not self.est.initialized():
            return (None, -1)
        
        if rescaling is None:
            rescaling = lambda R: 1
        if optimize_all_fn is None:
            optimize_all_fn = self.optimize_all
            
        # old throughput
        old_throughput = self.eval_throughput(cur_plan_config)
        if old_throughput <= 0:
            self.log(f"current plan config is infeasible\n{cur_plan_config}", level=logging.WARNING)
            return (None, -1)

        # optimize
        # new_resources may be reorded during optimize_all
        plans_and_throughputs, new_resources = optimize_all_fn(cur_plan_config, new_resources)
        
        # find the best plan config
        best = (None, -1)
        best_idx = -1
        for idx, (new_plan_config, new_throughput) in enumerate(plans_and_throughputs):
            new_acc_units_used = new_resources[:idx + 1]
            
            effective_speed_up = rescaling(new_acc_units_used) * \
                eval_effective_speed_up(pc1=cur_plan_config, pc2=new_plan_config, optim=self, 
                                        # thp1=old_throughput, thp2=new_throughput, 
                                        thp1=None, thp2=None, 
                                        fallback = -1)
                
            if effective_speed_up > best[1]:
                best_idx = idx
                best = (new_plan_config, effective_speed_up)
                
            # self.log(f"eval_effective_speed_up: idx={idx} >> {effective_speed_up * 100:.1f}%\n{new_plan_config}")
        
        self.log(f"eval_best_effective_speed_up: best={best_idx} >> {best[1] * 100:.1f}%\n{best[0]}")
        
        return best
                
    
    def load_balance(self, plan_config: PlanConfig, fractional: bool = False) -> PlanConfig:
        """
        Load balancing for the given accelerator mesh
        
        remember to set acc_mesh in plan_config to new one!!!
        global_batch_size, and total_num_layer are respected.
        """
        if plan_config is None:
            return None
        
        acc_mesh = plan_config.acc_mesh
        global_batch_size = plan_config.global_micro_batch_size()
        total_num_layer = plan_config.total_num_layer()
        
        # new_plan_config = copy.deepcopy(plan_config)
        new_plan_config = plan_config
        
        
        # DP load balance
        pp_deg = len(acc_mesh)
        lbsz_mesh = [None] * pp_deg
        for i in range(pp_deg):
            dp_mesh = acc_mesh[i]
            dp_deg = len(dp_mesh)
            
            tp_latencies = [
                sum(self.est._tp_latency(tp_mesh, 1, 1)[:2]) for tp_mesh in dp_mesh
            ]
            
            if fractional:
                inv = [1 / tp_latency for tp_latency in tp_latencies]
                sum_inv = sum(inv)
                lbszs = [global_batch_size * inv / sum_inv for inv in inv]
            else:
                lbszs = [0] * dp_deg
                heap = []
                for j in range(dp_deg):
                    heapq.heappush(heap, (tp_latencies[j] * (lbszs[j] + 1), j))
                    
                allocated_bsz = 0
                while allocated_bsz < global_batch_size:
                    _, j = heapq.heappop(heap)
                    lbszs[j] += 1
                    heapq.heappush(heap, (tp_latencies[j] * (lbszs[j] + 1), j))
                    allocated_bsz += 1

            lbsz_mesh[i] = lbszs
        new_plan_config.lbsz_mesh = lbsz_mesh


        # PP load balance
        dp_latencies = [
            sum(self.est._dp_latency(dp_mesh, lbszs, 1, 1)[:2]) for dp_mesh, lbszs in zip(acc_mesh, lbsz_mesh)
        ] # compute latency

        if fractional:
            inv = [1 / dp_latency for dp_latency in dp_latencies]
            sum_inv = sum(inv)
            layer_partition = [total_num_layer * inv / sum_inv for inv in inv]
        else:
            layer_partition = [0] * pp_deg
            heap = []
            for i in range(pp_deg):
                heapq.heappush(heap, (dp_latencies[i] * (layer_partition[i] + 1), i))
            
            allocated_layer = 0
            while allocated_layer < total_num_layer:
                _, i = heapq.heappop(heap)
                layer_partition[i] += 1
                heapq.heappush(heap, (dp_latencies[i] * (layer_partition[i] + 1), i))
                allocated_layer += 1
                    
        new_plan_config.layer_partition = layer_partition
        
        return new_plan_config
    
    def no_load_balance(self, plan_config: PlanConfig) -> PlanConfig:
        
        acc_mesh = plan_config.acc_mesh
        global_batch_size = plan_config.global_micro_batch_size()
        total_num_layer = plan_config.total_num_layer()
        
        # new_plan_config = copy.deepcopy(plan_config)
        new_plan_config = plan_config
        
        # DP load balance
        pp_deg = len(acc_mesh)
        lbsz_mesh = [None] * pp_deg
        for i in range(pp_deg):
            dp_mesh = acc_mesh[i]
            dp_deg = len(dp_mesh)
            lbsz_mesh[i] = [global_batch_size // dp_deg + (i < global_batch_size % dp_deg) for i in range(dp_deg)]
            
        new_plan_config.lbsz_mesh = lbsz_mesh

        # PP load balance
        layer_partition = [total_num_layer // pp_deg + (i < total_num_layer % pp_deg) for i in range(pp_deg)]
        new_plan_config.layer_partition = layer_partition
        
        return new_plan_config
    
    def optimize(
                self, 
                cur_plan_config: PlanConfig, 
                new_acc_units: AcceleratorMesh2D,
                ) -> PlanConfig:
        """
        optimize
            find the best plan config for the given plan config and new accelerator with dynamic programming based algo
        Args:
            it is considered as a list of minimal scheduler unit
        """
        
        # lb on current config
        cur_plan_config = self.load_balance(cur_plan_config)
                
        dp, _ = self.optimize_all(new_acc_units, cur_plan_config)
        
        return dp[-1][0]
    
    def grid_search(
                 self, 
                 cur_plan_config: PlanConfig, 
                 new_acc_units: AcceleratorMesh2D,
                 load_balancing: bool = True,
                 ) -> PlanConfig:
        """
        grid_search
            find the best plan config for the given plan config and new accelerator with grid search
        """
        cur_plan_config = self.load_balance(cur_plan_config)
                    
        plans = self._get_all_plan(cur_plan_config.acc_mesh, new_acc_units)
        best = (None, -1)
        
        for plan in plans:
            best = self._lb_compare_max(
                new_acc_mesh=plan, 
                cur_plan_config=cur_plan_config, 
                best_acc_mesh_and_throughput=best,
                load_balancing=load_balancing,
                fractional_lb = False
            )
        
        return best[0]
    
    def optimize_all(self, cur_plan_config: PlanConfig, new_acc_units: AcceleratorMesh2D, use_cpp_backend: bool =True) \
        -> Tuple[List[Tuple[PlanConfig, float]], AcceleratorMesh2D]:
        """
        Optimize all plans for the given plan config and new accelerator with dynamic programming based algorithm
        
        note that load balance is conducted on fraction during dp, but converted to integer load balance at last
        """
        if not self.est.initialized():
            return [], []
        
        # try Cpp backend first
        if optimize_all_cpp is not None and use_cpp_backend: 
            self.log("optimize_all: using Cpp backend")
            return optimize_all_cpp(
                cur_plan_config=cur_plan_config, 
                new_acc_units=new_acc_units, 
                rm=self.rm, 
                estimator_C=self.est.C,
                # log_file="cpp_backend.log"
            )
        else:
            self.log("optimize_all: using Python backend")
            

        acc_units_granularity = max(list(map(len, new_acc_units)) + [1])
        new_acc_units = sort_by_affinity(cur_plan_config.acc_mesh, new_acc_units, self.rm)
        
        new_acc_units = [[]] + new_acc_units # padding to 1 indexed
        n = len(new_acc_units)
        f = [(None, -1)] * n
        
        f[0] = (cur_plan_config, self.eval_throughput(cur_plan_config))
        
        cur_world_size = cur_plan_config.world_size()
        i_start = 0 if ENABLE_SELF_ADJUSTMENT else 1
        i_stop = min(n, max(default_n_acc_per_node, cur_world_size) // acc_units_granularity + 1) if ENABLE_LIMITED_EXPAND else n
        for i in range(i_start, i_stop):
            j_start = max(0, i - MAX_PREDECESSOR) if ENABLE_LIMITED_EXPAND else 0
            j_stop = i + 1 if ENABLE_SELF_ADJUSTMENT else i
            for j in range(j_start, j_stop):
                if f[j][0] is None:
                    continue
                
                units = new_acc_units[j + 1:i + 1]
                unit = flatten_mesh(units)
                
                tp_plan = self._get_best_tp_plan(f[j][0], unit)
                dp_plan = self._get_best_dp_plan(f[j][0], unit)
                pp_plan = self._get_best_pp_plan(f[j][0], units)
                
                tp_thp = self.eval_throughput(tp_plan)
                dp_thp = self.eval_throughput(dp_plan)
                pp_thp = self.eval_throughput(pp_plan)
                self.log(f"optimize_all:\n"
                         f"range=[{j+1},{i}] TP thp={tp_thp}:\n{tp_plan}\n"
                         f"range=[{j+1},{i}] DP thp={dp_thp}:\n{dp_plan}\n"
                         f"range=[{j+1},{i}] PP thp={pp_thp}:\n{pp_plan}\n")
                
                f[i] = self._compare_max(tp_plan, f[i])
                f[i] = self._compare_max(dp_plan, f[i])
                f[i] = self._compare_max(pp_plan, f[i])
                
            self.log(f"optimize_all: \nindex={i}, thp={f[i][1]}:\n{f[i][0]}\n")
        
        # # remove padding at 0
        # f.pop(0) 
        # new_acc_units.pop(0)
        
        # integer load balance
        real_plans = [
            self.load_balance(plan_config, fractional=False) for plan_config, _ in f
        ]
        f = [
            (plan, self.eval_throughput(plan)) for plan in real_plans
        ]
        
        return f, new_acc_units
    
    def log(self, msg: str, level=logging.DEBUG):
        if self.logger is not None:
            pattern_msg = f"JobOptimizer-{msg}"
            self.logger.log(level, pattern_msg)
    
    """
    Plan Enumeration Utility
    """
    def _get_next_tp_plan(self, 
                          acc_mesh: AcceleratorMesh3D, 
                          new_acc_unit: AcceleratorMesh1D, 
                          idx: Tuple[int, int]
                          ) -> tuple[AcceleratorMesh3D, Tuple[int, int]]:
        if idx is None:
            return None, None
        
        i, j = idx
        new_acc_mesh = copy.deepcopy(acc_mesh)
        new_acc_mesh[i][j].extend(new_acc_unit)
        
        j += 1
        if j >= len(new_acc_mesh[i]):
            i, j = i + 1, 0
        
        if i >= len(new_acc_mesh):
            idx = None
        else:
            idx = (i, j)
        
        return new_acc_mesh, idx
        
   
    def _get_next_dp_plan(self,                           
                          acc_mesh: AcceleratorMesh3D, 
                          new_acc_unit: AcceleratorMesh1D, 
                          idx: int):
        if idx is None:
            return None, None
        
        new_acc_mesh = copy.deepcopy(acc_mesh)
        new_acc_mesh[idx].append(new_acc_unit)
        
        idx += 1
        if idx >= len(new_acc_mesh):
            idx = None
            
        return new_acc_mesh, idx
        
    def _get_next_pp_plan(self,
                          acc_mesh: AcceleratorMesh3D, 
                          new_acc_unit: AcceleratorMesh1D, 
                          ):
        new_acc_mesh = copy.deepcopy(acc_mesh)
        new_acc_mesh.append([new_acc_unit])
        
        return new_acc_mesh, None
    
    def _get_all_plan_one_step(self,
                      acc_mesh: AcceleratorMesh3D, 
                      new_acc_unit: AcceleratorMesh1D, 
                      ) -> List[AcceleratorMesh3D]:
        """
        Get all possible plans for the given accelerator mesh and new accelerator (considered as minimal scheduler unit)
        """
       
        plans = []
        idx = (0, 0)
        while idx != None:
            plan, idx = self._get_next_tp_plan(acc_mesh, new_acc_unit, idx)
            plans.append(plan)
            
        idx = 0
        while idx != None:
            plan, idx = self._get_next_dp_plan(acc_mesh, new_acc_unit, idx)
            plans.append(plan)
            
        plan, idx = self._get_next_pp_plan(acc_mesh, new_acc_unit)
        plans.append(plan)
        
        return plans
    
    def _get_all_plan(self,
                      acc_mesh: AcceleratorMesh3D,
                      new_acc_units: AcceleratorMesh2D 
                      ) -> List[AcceleratorMesh3D]:
        """    
        Get all possible plans for the given accelerator mesh and new accelerator list 
        # each element in list is considered as minimal scheduler unit
        BFS style grid search
        """
        
        plans = [acc_mesh]
        
        for new_acc_unit in new_acc_units:
            new_plans = [] 
            for plan in plans:
                new_plans.extend(
                    self._get_all_plan_one_step(plan, new_acc_unit)
                )
            plans = new_plans
        
        if not OVERRIDE_CHECK_VALID_MESH:
            valid_plans = [ plan for plan in plans if check_valid_acc_mesh(plan) ]
            return valid_plans
        else:
            return plans
        
    @staticmethod
    def _to_target_TP_degree(accs: AcceleratorMesh1D, tp_deg: int) -> AcceleratorMesh2D:
        """
        Rearrange a list of accs into 2 2D mesh with given tp_deg. All tp must be intra-node
        """
        group_by_node_id: Dict[int, List[Accelerator]] = {}
        for acc in accs:
            if acc.node_id not in group_by_node_id:
                group_by_node_id[acc.node_id] = []
            group_by_node_id[acc.node_id].append(acc)
        new_mesh2D = []
        for node_id, accs_on_node in group_by_node_id.items():
            if len(accs_on_node) % tp_deg != 0:
                return None
            accs_on_node.sort(key=lambda acc: acc.local_id if acc.local_id is not None else 0)
            for i in range(0, len(accs_on_node), tp_deg):
                new_mesh2D.append(accs_on_node[i:i + tp_deg])
        return new_mesh2D
    
    def _half_TP_degree(self, cur_plan_config: PlanConfig) -> PlanConfig:
        """
        Idea: if we add new accelerators into the plan, memory pressure will likely reduce. 
             therefore, there are chances to half tp degree (and double dp degree) to trade for speed
        """
        if cur_plan_config is None:
            return None
        
        best = (cur_plan_config, self.eval_throughput(cur_plan_config))
        
        tp_degs = [ len(mesh2D[0]) for mesh2D in cur_plan_config.acc_mesh ]
        tp_degs_order = sorted(range(len(tp_degs)), key=lambda i: tp_degs[i], reverse=True)
        
        for pp_idx in tp_degs_order:
            acc_mesh = copy.deepcopy(best[0].acc_mesh)
            mesh2D = acc_mesh[pp_idx]
            cur_tp_deg = len(mesh2D[0])
            if cur_tp_deg <= 1:
                continue
            new_mesh2D = self._to_target_TP_degree(flatten_mesh(mesh2D), cur_tp_deg // 2)
            if new_mesh2D is None:
                continue # ...? should never happen. half TP degree is always possible
            acc_mesh[pp_idx] = new_mesh2D
            best = self._lb_compare_max(
                new_acc_mesh=acc_mesh, 
                cur_plan_config=cur_plan_config, 
                best_acc_mesh_and_throughput=best,
                load_balancing=True, 
                fractional_lb=False # half TP degree may have severe rounding issue for small batch size, so set frational lb to False here.
            )
            
        return best[0]

        
    def _get_best_tp_plan(self, 
                          cur_plan_config: PlanConfig, 
                          new_acc_unit: AcceleratorMesh1D) -> PlanConfig:
        """
        Get the best tp plan for the given plan config and new accelerator
        """
        # if len(new_acc_unit) > default_n_acc_per_node or len(new_acc_unit) == 0:
        #     return None

        # idx = (0, 0)
        # best = (None, -1)
        # acc_mesh = cur_plan_config.acc_mesh
        # while idx != None:
        #     new_plan, idx = self._get_next_tp_plan(acc_mesh, new_acc_unit, idx)
        #     best = self._lb_compare_max(new_plan, cur_plan_config, best, fractional_lb=True)
            
        # return best[0]

        """
        Improved algorithm: attempt to double the TP degree of a PP stage
        """

        best = (None, -1)
        cur_acc_mesh = cur_plan_config.acc_mesh
        for pp_idx, mesh2D in enumerate(cur_acc_mesh):
            cur_acc_unit = flatten_mesh(mesh2D)
            # if len(cur_acc_unit) > len(new_acc_unit):
            #     continue
            
            cur_tp_deg = len(mesh2D[0])
            new_mesh2D = self._to_target_TP_degree(cur_acc_unit + new_acc_unit, cur_tp_deg * 2)          

            if new_mesh2D is None:
                continue
            
            new_acc_mesh = copy.deepcopy(cur_acc_mesh)
            new_acc_mesh[pp_idx] = new_mesh2D
            
            best = self._lb_compare_max(new_acc_mesh, cur_plan_config, best, load_balancing=True, fractional_lb=ENABLE_FRACTIONAL_LB)
        
        return self._half_TP_degree(best[0])
    
    def _get_best_dp_plan(self,
                          cur_plan_config: PlanConfig, 
                          new_acc_unit: AcceleratorMesh1D) -> PlanConfig:
        """
        Get the best dp plan for the given plan config and new accelerator
        """
        # idx = 0
        # best = (None, -1)
        # acc_mesh = cur_plan_config.acc_mesh
        # while idx != None:
        #     new_plan, idx = self._get_next_dp_plan(acc_mesh, new_acc_unit, idx)
        #     best = self._lb_compare_max(new_plan, cur_plan_config, best, fractional_lb=True)
        # return best[0]
        
        # keep tp degree constistent, attempt to add to a stage
        best = (None, -1)
        cur_acc_mesh = cur_plan_config.acc_mesh
        for pp_idx, mesh2D in enumerate(cur_acc_mesh):
            cur_acc_unit = flatten_mesh(mesh2D)
            cur_tp_deg = len(mesh2D[0])
            if len(new_acc_unit) % cur_tp_deg != 0:
                continue
            
            new_mesh2D = self._to_target_TP_degree(cur_acc_unit + new_acc_unit, cur_tp_deg)
            if new_mesh2D is None:
                continue

            new_acc_mesh = copy.deepcopy(cur_acc_mesh)            
            new_acc_mesh[pp_idx] = new_mesh2D
            
            best = self._lb_compare_max(new_acc_mesh, cur_plan_config, best, load_balancing=True, fractional_lb=ENABLE_FRACTIONAL_LB)
            
        return self._half_TP_degree(best[0])
    
    
    def _get_best_pp_plan(self,
                          cur_plan_config: PlanConfig, 
                          new_acc_units: AcceleratorMesh2D) -> PlanConfig:
        """
        Get the best pp plan for the given plan config and new accelerator
        """        
        
        # # utility to merge slowest units
        # def merge(units: AcceleratorMesh2D) -> bool:
        #     if len(units) <= 1:
        #         return False
            
        #     # attempt to merge two slowest mergable units
        #     def mergeable(acc_unit1: AcceleratorMesh1D, acc_unit2: AcceleratorMesh1D):
        #         if len(acc_unit1) != len(acc_unit2):
        #             return False
                
        #         if acc_unit1[0].acc_type_name() != acc_unit2[0].acc_type_name():
        #             return False
                
        #         return True
            
        #     def bandwidth(acc_unit1: AcceleratorMesh1D, acc_unit2: AcceleratorMesh1D) -> float:
        #         bw_sum = 0
        #         for acc1 in acc_unit1:
        #             for acc2 in acc_unit2:
        #                 bw_sum += self.rm.get_bandwidth(acc1, acc2)
        #         return bw_sum
            
        #     m = len(units)
                        
        #     TFLOPS_sum = [
        #         self.rm._sumTFLOPS(acc_unit) for acc_unit in units
        #     ]
        #     # slowest first
        #     TFLOPS_rank = [i for i in range(m)]
        #     TFLOPS_rank.sort(
        #         key=lambda i: TFLOPS_sum[i]
        #     )

        #     i = 0            
        #     I = TFLOPS_rank[i]
        #     candidates = []
        #     for j in range(i + 1, m):
        #         J = TFLOPS_rank[j]
        #         if mergeable(units[I], units[J]):
        #             candidates.append(J)
        #     if not candidates:
        #         return False
            
        #     # best candidate
        #     J = max(candidates, key=lambda J: bandwidth(units[I], units[J]))
            
        #     # merge
        #     units[I].extend(units[J])
        #     units.pop(J)
            
        #     return True

        # # actual algorithm using merge
        # units = copy.deepcopy(new_acc_units)
        # acc_mesh = copy.deepcopy(cur_plan_config.acc_mesh)
        # acc_mesh.append(units)
        # best = self._lb_compare_max(acc_mesh, cur_plan_config, (None, -1))
        
        # while merge(units):
        #     last_best = best
        #     acc_mesh[-1] = units
        #     best = self._lb_compare_max(acc_mesh, cur_plan_config, best)
        #     if best[1] > 0 and last_best[1] >= best[1]: # already feasible and no improvement
        #         return last_best[0]
            
        # return best[0]
        
        """
        Improved Algorithm: doulbe TP degree till feasible and no improvement
        """
        
        cur_tp_deg = 1
        new_acc_unit = flatten_mesh(new_acc_units) # only for legacy interface compatibility
        if not new_acc_unit:
            return None
        
        new_stage_mesh2D = self._to_target_TP_degree(
            new_acc_unit, cur_tp_deg
        )
        new_acc_mesh = copy.deepcopy(cur_plan_config.acc_mesh)
        new_acc_mesh.append(new_stage_mesh2D)
        best = self._lb_compare_max(new_acc_mesh, cur_plan_config, (None, -1), load_balancing=True, fractional_lb=ENABLE_FRACTIONAL_LB)

        while True:
            cur_tp_deg *= 2    
            new_stage_mesh2D = self._to_target_TP_degree(new_acc_unit, cur_tp_deg)
            
            if new_stage_mesh2D is None:
                break
            
            last_best = best
            new_acc_mesh[-1] = new_stage_mesh2D # we have deepcopy in lb compare max, it's safe to modify in place
            best = self._lb_compare_max(new_acc_mesh, cur_plan_config, best, load_balancing=True, fractional_lb=ENABLE_FRACTIONAL_LB)
            if best[1] > 0 and last_best[1] >= best[1]: # already feasible and no improvement
                break
        
        return self._half_TP_degree(best[0])
    
    """
    Other Utils
    """
    
    def _compare_max(
        self,
        new_plan_config: PlanConfig,
        best_plan_config_and_throughput: Tuple[PlanConfig, float] = (None, -1),
    ) -> Tuple[PlanConfig, float]:
        """
        Compare the new plan config with the best plan config
        """
        
        best_plan_config, best_throughput = best_plan_config_and_throughput
        
        new_throughput = self.eval_throughput(new_plan_config)
        
        if best_throughput * (1+EPS) < new_throughput:
            return new_plan_config, new_throughput
        else:
            return best_plan_config, best_throughput
        
    def _lb_compare_max(
        self,
        new_acc_mesh: AcceleratorMesh3D,
        cur_plan_config: PlanConfig,
        best_acc_mesh_and_throughput: Tuple[PlanConfig, float] = (None, -1),
        load_balancing: bool = True,
        fractional_lb: bool = True,
    ) -> Tuple[PlanConfig, float]:
        """
        Compare the new plan config with the best plan config after lb.
        also check for validity of mesh
        """
        if not OVERRIDE_CHECK_VALID_MESH and not check_valid_acc_mesh(new_acc_mesh):
            return best_acc_mesh_and_throughput
            
        new_plan_config = copy.deepcopy(cur_plan_config)
        new_plan_config.acc_mesh = copy.deepcopy(new_acc_mesh)
        if load_balancing:
            new_plan_config = self.load_balance(new_plan_config, fractional=fractional_lb)
        else:
            new_plan_config = self.no_load_balance(new_plan_config)
        
        return self._compare_max(new_plan_config, best_acc_mesh_and_throughput)
    
    
  

    
"""
utils to partition acc mesh1D to mesh2D for scheduling
"""
def partition_acc(new_acc_unit: AcceleratorMesh1D, minimal_decision_num_acc: int) -> AcceleratorMesh2D:
    """
    Partition the new accelerator by nodes with each no more than minimal_decision_num_acc
    """
    new_acc_dict = { acc.node_id: [] for acc in new_acc_unit }
    for acc in new_acc_unit:
        new_acc_dict[acc.node_id].append(acc)
        
    new_acc_units = []
    
    for node_id, acc_list in new_acc_dict.items():
        if minimal_decision_num_acc != -1 and len(acc_list) > minimal_decision_num_acc:
            for i in range(0, len(acc_list), minimal_decision_num_acc):
                new_acc_units.append(acc_list[i:i + minimal_decision_num_acc])
        else:
            new_acc_units.append(acc_list)
    
    return new_acc_units


"""
evaluate effective speed up of pc2 over pc1
"""
def eval_effective_speed_up(pc1: PlanConfig, pc2: PlanConfig, optim: JobOptimizer, thp1: float = None, thp2: float = None, fallback: float = -1.0) -> float:
    
    if thp1 is None:
        thp1 = optim.eval_throughput(pc1)
    if thp2 is None:
        thp2 = optim.eval_throughput(pc2)
    
    if thp1 <= 0 or thp2 <= 0:
        return fallback # not measurable, likely due to optim is not initialized after reconfiguration / or pc is None?
    if abs(thp1 - thp2) / min(thp1, thp2) < ABSOLUTE_IMPROVEMENT_MARGIN:
        return 0.0

    tflops1 = optim.rm._sumTFLOPS(pc1.acc_mesh)
    tflops2 = optim.rm._sumTFLOPS(pc2.acc_mesh)
    
    if abs(tflops1 - tflops2) < EPS:
        if thp1 < thp2:
            return 1.
        else:
            return -1.
    
    return (thp2 - thp1) / (thp1 * (tflops2 / tflops1 - 1))

def sort_by_affinity(cur_acc_mesh: AcceleratorMesh3D, new_acc_units: AcceleratorMesh2D, rm: ResourceManager) -> List[AcceleratorMesh1D]:
    """
    Reorder new_acc_units by network topology affinity
    key : (bandwidth, -relative_rank, -node_id, -local_id) 
    bandwidth ensures accs with better network affinity are used
    relative_rank ensures accs that can increase TP degree of a PP stage are grouped continuously
    node_id, local_id ensures that the accs in same node are grouped together
    """
    
    flatten_cur_acc_mesh = flatten_mesh(cur_acc_mesh)
    def get_sort_key(new_acc_unit: AcceleratorMesh1D) -> Tuple[float, int, int]:
        new_acc = new_acc_unit[0]
        
        relative_rank = -1
        max_bandwidth = 0
        for i, cur_acc in enumerate(flatten_cur_acc_mesh):    
            if rm.get_bandwidth(new_acc, cur_acc) > max_bandwidth:
                max_bandwidth = rm.get_bandwidth(new_acc, cur_acc)
            if cur_acc.node_id == new_acc.node_id:
                relative_rank = i
            
        return (
            max(rm.get_bandwidth(new_acc, cur_acc) for cur_acc in flatten_cur_acc_mesh),
            -relative_rank,
            -new_acc.node_id if new_acc.node_id is not None else 0,
            -new_acc.local_id if new_acc.local_id is not None else 0
        )
    
    affinity_score = [
        get_sort_key(new_acc_unit) for new_acc_unit in new_acc_units
    ]
    affinity_order = sorted(range(len(affinity_score)), key=lambda i: affinity_score[i], reverse=True)
    new_acc_units = [new_acc_units[i] for i in affinity_order]
    
    return new_acc_units