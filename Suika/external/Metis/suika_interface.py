"""
  This file is not included in the original Metis, and serves as interface to Suika framework
  optimize function is the entry point to Suika framework, and is adapted from __main__ code in cost_het_cluster.py
"""

from Suika.external.Metis.data_loader import ProfileDataLoader
from Suika.external.Metis.model import device_group
from Suika.external.Metis.model.cost_estimator import HeteroCostEstimator
from Suika.external.Metis.model.activation_parameter import GPTActivationAndParam
from Suika.external.Metis.model.load_balancer import DataLoadBalancer, LayerLoadBalancer
from Suika.external.Metis.gpu_cluster import GPUCluster
from Suika.external.Metis.search_space.plan import InterStagePlan
from Suika.external.Metis.utils import MetisModelConfig, metis_logger
from Suika.external.Metis.cost_het_cluster import cost_het_cluster


from Suika.cluster.resource import Accelerator, Node, ResourceManager, default_n_acc_per_node
from Suika.cluster.estimator import AcceleratorMesh2D, sort_by_affinity
from Suika.cluster.plan_config import PlanConfig
from Suika.utils import flatten_mesh
from Suika.cluster.job import Job

from typing import List, Dict, Optional, Tuple
import os
import copy

"""
Convert Metis plan descriptor to Suika plan descriptor

Discussion about Rank to Node mapping during conversion:

  Metis plan descriptor specify
  1. node sequence (ordering of device type)
  2. device groups (stage-wise device group size)
  which derives a rank-to-device_type mapping; rank-to-node mapping is not uniquely determined.
  
  While there is rank-to-node mapping in ClusterBandwidth, it is not used for TP communication latency estimation.
  
  We implement two rank-to-node mapping strategy: (toggled by respect_rank_node_mapping)
    1. respect the rank-to-node mapping as defined in ClusterBandwidth, and enforce each TP group to be on the same node.
    2. only respect rank-to-device_type mapping, and enforce each TP group to be on the same node. 
   
    
  This is reasonable, because Metis's performance estimation evaluates TP latency by (likely single-node) profiling data, and makes similar assumption in Section 4.2.1 in paper.

  The 'same node for TP' requirement may yield infeasible allocation, especially in increamental reconfiguration scenario, say:
      - 2 nodes each with 2 GPUs are provided
      - Metis decides 4TP is the best plan
      - the ResourceManager is not able to allocate 4 GPUs on the same node
      
  To address infeasible allocation, we attempt to convert Metis plan and allocate with ResourceManager.
  If failed, we try next best plan until feasible allocation is found or max_attemps expires. (implemented in optimize function)

  (Are we missing somewhere that Metis has excluded inter-node TP plans?)
"""
def convert_Metis_plan_to_Suika_plan(
  metis_plan: Tuple, # one entry from cost_het_cluster
  gpu_cluster: GPUCluster, # for device placement
  gbs: int, profile_data: Dict, model_config: MetisModelConfig,  # for data load balancer
  respect_rank_node_mapping: bool = False
  ) -> Optional[PlanConfig]:
  # node_sequence(0), device_groups(1), strategies(dp_deg, tp_deg)(2), batches(number of batch)(3), layer_partition(4), num_repartition(5), cost(6)
  node_sequence = metis_plan[0]
  device_groups = metis_plan[1]
  strategies = metis_plan[2]
  batches = metis_plan[3]
  accumulative_layer_partition = metis_plan[4]
    
  # get acc_mesh
  # adapated from cluster_bandwidth.py
  def _get_device_placement(gpu_cluster: GPUCluster, node_sequence: Optional[List[str]] = None) -> Tuple[Dict, Dict]:
      rank_node_map_absolute = dict()
      
      nodes = list(gpu_cluster.nodes.items())
      
      if node_sequence:
          idx_node_sequence = {
              node_id: node_sequence.index(node.acc_type_name()) for node_id, node in nodes
          }
          nodes = sorted(nodes, key= lambda x: idx_node_sequence[x[0]])

      counter = 0
      devices = []
      for _, (node_id, node) in enumerate(nodes):
          for _ in range(node.acc_num):
              rank_node_map_absolute[counter] = node_id
              devices.append(node.acc_type_name())
              counter += 1

      return rank_node_map_absolute, devices
  
  rank_to_node, devices = _get_device_placement(gpu_cluster, node_sequence)

  acc_mesh = []
  pp_deg = len(device_groups)
  for pp_id in range(pp_deg):
    start_rank = sum(device_groups[:pp_id])
    end_rank = sum(device_groups[:pp_id + 1])
    
    start_node_id = sum(dp_deg for dp_deg, tp_deg in strategies[:pp_id])
    
    dp_deg, tp_deg = strategies[pp_id]
    
    def get_rank(dp_id, tp_id):
      return start_rank + tp_id + dp_id * tp_deg
    
    # dp x tp acc_mesh, where each tp group is on the same node
    stage_acc_mesh = [
      [
        Accelerator(
          acc_type=devices[get_rank(dp_id, tp_id)],
          node_id= \
            rank_to_node[get_rank(dp_id, tp_id)] if respect_rank_node_mapping else start_node_id + dp_id,
          local_id=tp_id
        )
        for tp_id in range(tp_deg)
      ]
      for dp_id in range(dp_deg)
    ]

    # check if TP group is on the same node
    for dp_id in range(dp_deg):
      if respect_rank_node_mapping: 
        if not all(rank_to_node[get_rank(dp_id, tp_id)] == rank_to_node[get_rank(dp_id, 0)] for tp_id in range(tp_deg)):
          metis_logger.warning('TP group is not on the same node')
          return None
      else:
        if not all(devices[get_rank(dp_id, tp_id)] == devices[get_rank(dp_id, 0)] for tp_id in range(tp_deg)):
          metis_logger.warning('TP group is not homogeneous, and is impossible to allocate on the same node')
          return None

    acc_mesh.append(stage_acc_mesh)
    
  # lbsz_mesh
  lbsz_mesh = []
  for pp_id in range(pp_deg):
    start_rank = sum(device_groups[:pp_id])
    end_rank = sum(device_groups[:pp_id + 1])
    devices_stage = devices[start_rank: end_rank]
    intra_strategy = strategies[pp_id]
    
    data_load_balancer = DataLoadBalancer(profile_data, model_config)
    hetero_bs = data_load_balancer.partition_data(device_types=devices_stage, 
                                                  intra_strategy=intra_strategy, 
                                                  bs=gbs // batches)
    lbsz_mesh.append(hetero_bs)
  
  diff_layer_partition = [
    accumulative_layer_partition[i + 1] - accumulative_layer_partition[i] for i in range(len(accumulative_layer_partition) - 1)
  ]
  num_micro_batch = batches
  
  plan_config = PlanConfig(
    acc_mesh=acc_mesh,
    lbsz_mesh=lbsz_mesh,
    layer_partition=diff_layer_partition,
    num_micro_batch=num_micro_batch,
  )
  # metis_logger.debug(f"? metis_plan {metis_plan}, \nsuika_plan {plan_config}")
  
  return plan_config

"""
@EXPORT

Estimate PlanConfig's throughput using Metis performance model

Output: Latency in seconds
"""
def Metis_estimate(plan_config: PlanConfig, model_config: MetisModelConfig, rm: ResourceManager) -> float:  
  if plan_config is None:
    return float("inf")
  
  gbs = plan_config.global_batch_size()
  n_mb = plan_config.num_micro_batch 
  
  # prepare model_config
  if not isinstance(model_config, MetisModelConfig):
    model_config = MetisModelConfig.from_SuikaConfig(model_config)
  
  # prepare GPUCluster
  all_resources = flatten_mesh(plan_config.acc_mesh) if plan_config is not None else []
  
  node_dict: Dict[int, Node] = {}
  for acc in all_resources:
    acc: Accelerator
    node_id = acc.node_id
    if node_id not in node_dict:
      node_dict[node_id] = Node(acc_type=acc.acc_type_name(), node_id=node_id, acc_num=0)
    node_dict[node_id].acc_num += 1
  node_list = list(node_dict.values())
  
  gpu_cluster = GPUCluster(rm, node_list)
  
  # prepare ProfileDataLoader
  profile_path = f"{os.path.dirname(os.path.abspath(__file__))}/profile_data/{model_config.model_name}"
  data_loader = ProfileDataLoader(profile_path)
  profile_data, _, max_tp, max_bs = data_loader.load_profile_data_all()
  

  assert len(profile_data.keys()) > 0, f'No profile data found for {model_config.model_name}'
  
  # prepare solver utilities
  model_volume = GPTActivationAndParam(
                                      model_config=model_config, 
                                      model_params=profile_data['model']['parameters']
                                      )
  cost_estimator = HeteroCostEstimator(
                                      profile_data=profile_data, 
                                      model_config=model_config, 
                                      model_volume=model_volume, 
                                      gpu_cluster=gpu_cluster)


  # Strategies
  strategies = [] # list of (dp_deg, tp_deg)
  for stage_acc_mesh in plan_config.acc_mesh:
    dp_deg = len(stage_acc_mesh)
    tp_deg = len(stage_acc_mesh[0])
    strategies.append((dp_deg, tp_deg))

  # Layer partition
  layer_partition = [0]
  for n_layer in plan_config.layer_partition:
    layer_partition.append(layer_partition[-1] + n_layer)
    
  # InterStagePlan
  node_sequence = [node.acc_type_name() for node in gpu_cluster.nodes.values()]
  device_groups = [dp_deg * tp_deg for dp_deg, tp_deg in strategies]
  plan = InterStagePlan(ns_idx=0, # does not matter
                 node_sequence=node_sequence,
                 dg_idx=0, # does not matter
                 device_groups=device_groups,
                 num_stage=len(layer_partition) - 1,
                 batches=n_mb,
                 gbs=gbs
                )
  
  # Rank device map
  rank_device_map = {}
  for pp_id in range(len(plan_config.acc_mesh)):
    dp_mesh = plan_config.acc_mesh[pp_id]
    for dp_id in range(len(dp_mesh)):
      tp_mesh = dp_mesh[dp_id]
      for tp_id in range(len(tp_mesh)):
        acc = tp_mesh[tp_id]
        rank = plan_config.rank_mesh[pp_id][dp_id][tp_id]
        rank_device_map[rank] = acc.acc_type_name()
        
  # get cost
  cost = cost_estimator.get_cost(plan, strategies, layer_partition, rank_device_map)
  
  return cost

"""
@EXPORT

args:
  cur_plan_config: PlanConfig describing current execution plan (not None)
  new_resources: List of new acceleorators (physical) to be added to the current job
  rm: ResourceManager object of entire cluster
    (masking of rm to resources of this job is done under the hood without user's knowledge)

return:
  (PlanConfig, throughput) or (None, -1)
  plan config is forced to use all new resources
  
note:
  1.though the interface is intended for incremental resource allocation,
    if user_plan_config's global_batch_size and total_num_layer is valid, 
      cur_plan_config is None,
    the interface can find an initial plan for the job with new_resources. 
    (literally, the original design purpose of Metis)
  2. profile_data should be placed under
    |-./profile_data/{model_config.model_name}/...
    (the root dir is the same as this file)
  3. 'model_config' (ModelConfig instance) or 'model_strategy' (ModelStrategy instance) must be set in job.other_context; model_config, and model_name are inferred from that.
  
"""
def Metis_optimize(cur_plan_config: PlanConfig, model_config: MetisModelConfig, new_resources: List[Accelerator], rm: ResourceManager,  respect_rank_node_mapping: bool) -> Tuple[PlanConfig, float]:
  # prepare basic plan info
  gbs = cur_plan_config.global_batch_size()
  num_layers = cur_plan_config.total_num_layer()
  
  # prepare model_config
  if not isinstance(model_config, MetisModelConfig):
    model_config = MetisModelConfig.from_SuikaConfig(model_config)
  
  # prepare GPUCluster
  cur_resources = flatten_mesh(cur_plan_config.acc_mesh) if cur_plan_config is not None else []
  new_resources = flatten_mesh(new_resources)
  all_resources = cur_resources + new_resources
  
  node_dict: Dict[int, Node] = {}
  for acc in all_resources:
    acc: Accelerator
    node_id = acc.node_id
    if node_id not in node_dict:
      node_dict[node_id] = Node(acc_type=acc.acc_type_name(), node_id=node_id, acc_num=0)
    node_dict[node_id].acc_num += 1
  node_list = list(node_dict.values())
  
  gpu_cluster = GPUCluster(rm, node_list)
  
  # prepare ProfileDataLoader
  profile_path = f"{os.path.dirname(os.path.abspath(__file__))}/profile_data/{model_config.model_name}"
  data_loader = ProfileDataLoader(profile_path)
  profile_data, _, max_tp, max_bs = data_loader.load_profile_data_all()
  

  assert len(profile_data.keys()) > 0, f'No profile data found for {model_config.model_name}'
  
  # prepare solver utilities
  model_volume = GPTActivationAndParam(
                                      model_config=model_config, 
                                      model_params=profile_data['model']['parameters']
                                      )
  cost_estimator = HeteroCostEstimator(
                                      profile_data=profile_data, 
                                      model_config=model_config, 
                                      model_volume=model_volume, 
                                      gpu_cluster=gpu_cluster)
  layer_load_balancer = LayerLoadBalancer(
                                      gpu_cluster=gpu_cluster, 
                                      profile_data=profile_data, 
                                      model_config=model_config, 
                                      gbs=gbs)
  
  
  # node_sequence(0), device_groups(1), strategies(dp_deg, tp_deg)(2), batches(number of batch)(3), layer_partition(4), num_repartition(5), cost(6)
  estimate_costs = cost_het_cluster(
                            gbs=gbs,
                            num_layers=num_layers, 
                            gpu_cluster=gpu_cluster, 
                            profile_data=profile_data, 
                            model_config=model_config, 
                            cost_estimator=cost_estimator, 
                            layer_load_balancer=layer_load_balancer,
                            max_profiled_tp_degree=max_tp,
                            max_profiled_batch_size=max_bs
                          )
  
  sorted_result = sorted(estimate_costs, key=lambda kv: kv[6])
  
  # iterate through sorted_result and find the first best plan with feasible TP allocation (for detail, see comments at convert_Metis_plan_to_Suika_plan)
  v_rm = copy.deepcopy(rm)
  
  v_rm.free(cur_resources)
  
  best_plan = (None, -1)
  max_alloc_attemps = 10
  for idx, result in enumerate(sorted_result):
    new_plan_config = convert_Metis_plan_to_Suika_plan(
      metis_plan=result,
      gpu_cluster=gpu_cluster,
      gbs=gbs,
      profile_data=profile_data,
      model_config=model_config,
      respect_rank_node_mapping=respect_rank_node_mapping
    )
    
    if new_plan_config is not None and \
        v_rm.test_allocate(new_plan_config.acc_mesh, mask=all_resources):
      # feasible allocation found
      new_acc_mesh = v_rm.allocate(new_plan_config.acc_mesh, mask=all_resources, virtual_test=True)
      new_plan_config = new_plan_config.instantiate(new_acc_mesh)
      
      cost = result[6] / 1000 # convert mircosecond to second
      best_plan = (new_plan_config, 1/cost)
      break
    if idx >= max_alloc_attemps:
      break
      
  return best_plan

def Metis_optimize_all(
  job: Job,
  cur_plan_config: PlanConfig,
  new_acc_units: AcceleratorMesh2D,
  time_budget: int, # in seconds
  rm: ResourceManager = None
) -> Tuple[List[Tuple[PlanConfig, float]], AcceleratorMesh2D]:
  if cur_plan_config is None:
    metis_logger.warning(f"job ({job}) cur_plan_config is None")
    return [], []
  
  # model_config
  try:  
    if "model_strategy" in job.other_context:
      model_config = job.other_context["model_strategy"].model_config
    else:
      model_config = job.other_context["model_config"]
      
    model_config = MetisModelConfig.from_SuikaConfig(model_config)    
  except KeyError:
    raise KeyError(f"Metis: model_config/model_strategy not set in job.other_context ({job})")
  
  # rm
  if rm is None:
    rm = job.job_optimizer.est.rm
  v_rm = copy.deepcopy(rm)
  
  # new_acc_units
  new_acc_units = sort_by_affinity(cur_acc_mesh=cur_plan_config.acc_mesh, new_acc_units=new_acc_units, rm=v_rm)
  new_acc_units = [[]] + new_acc_units
  
  n = len(new_acc_units)
  optimize_results = [(None, -1)] * n
  
  # optimize with time budget, using mp
  import multiprocessing, time
  def optimize_worker(new_resources: List[Accelerator], queue: multiprocessing.Queue):
    try:
      plan_config, throughput = Metis_optimize(
        cur_plan_config=cur_plan_config,
        model_config=model_config,
        new_resources=new_resources,
        rm=v_rm,
        respect_rank_node_mapping=True,
      )
      queue.put((plan_config, throughput))
    except Exception as e:
      metis_logger.error(f"Error in optimize_worker {i}: {e}")
      queue.put((None, -1.1)) # differ from -1 for debug
  
  # now = time.time()
  # start_time = now
  # new_resources = []
  # queue = multiprocessing.Queue()
  # for i, acc_unit in enumerate(new_acc_units):
  #   now = time.time()
  #   remaining_time_budget = time_budget - (now - start_time)
    
  #   new_resources += acc_unit
  #   worker_proc = multiprocessing.Process(
  #     target=optimize_worker,
  #     args=(new_resources, queue)
  #   )
  #   worker_proc.start()
    
  #   worker_proc.join(timeout=remaining_time_budget)
    
  #   if worker_proc.is_alive():
  #     worker_proc.terminate()
  #     metis_logger.info(f"Metis optimize all timeout after {time_budget} seconds {i}/{n} cases")
  #     break
  #   else:
  #     optimize_results[i] = queue.get()
  
  # parallelize every worker
  worker_procs = []
  queues = [multiprocessing.Queue() for _ in range(n)]
  new_resources = []
  expansion_limit = max([cur_plan_config.world_size(), default_n_acc_per_node, len(new_acc_units)])
  for i, acc_unit in enumerate(new_acc_units):
    if i + 1 > expansion_limit: # enable limited expansion, avoid overwhelming CPU cores
      break
    new_resources += acc_unit
    worker_proc = multiprocessing.Process(
      target=optimize_worker,
      args=(new_resources, queues[i])
    )
    worker_proc.start()

    worker_procs.append(worker_proc)

  now = time.time()
  start_time = now
  timeout_list = []
  for i, worker_proc in enumerate(worker_procs):
    now = time.time()
    remaining_time_budget = time_budget - (now - start_time)
    
    worker_proc.join(timeout=remaining_time_budget)
    if worker_proc.is_alive():
      worker_proc.terminate()
      timeout_list += [i]
    else:
      optimize_results[i] = queues[i].get()

  if timeout_list:
    metis_logger.info(f"{job}: Metis_optimize_all timeout for {len(timeout_list)} / {expansion_limit} cases {timeout_list} after {time_budget} seconds")
  else:
    metis_logger.info(f"{job}: Metis_optimize_all all {expansion_limit} cases solved within {time_budget} seconds!")

  return optimize_results, new_acc_units  