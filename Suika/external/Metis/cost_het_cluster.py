# Metis/cost_het_cluster.py
# Copyright 2024 Samsung Electronics Co., Ltd. All Rights Reserved

# @adaptation: remove args, use explict parameters from hyperparam.py and gbs/num_layers parameters

from typing import Dict, List, Tuple
import sys
import os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Suika.external.Metis.data_loader import ProfileDataLoader
from Suika.external.Metis.model.cost_estimator import HeteroCostEstimator
from Suika.external.Metis.model.activation_parameter import GPTActivationAndParam
from Suika.external.Metis.model.device_group import StagePerformance
from Suika.external.Metis.model.load_balancer import LayerLoadBalancer
from Suika.external.Metis.search_space.plan import IntraStagePlanGenerator, InterStagePlanGenerator
from Suika.external.Metis.gpu_cluster import GPUCluster
from Suika.external.Metis.utils import MetisModelConfig, metis_logger

import Suika.external.Metis.hyperparam as args

def cost_het_cluster(gbs:int, num_layers: int, gpu_cluster: GPUCluster, profile_data: Dict, model_config: MetisModelConfig,
                     cost_estimator: HeteroCostEstimator, layer_load_balancer:LayerLoadBalancer, max_profiled_tp_degree: int, max_profiled_batch_size: int) -> List[Tuple]:

    estimate_costs = []
    for inter_stage_plan in InterStagePlanGenerator(device_types=set(gpu_cluster.get_device_types()),
                                                    num_devices=gpu_cluster.get_total_num_devices(),
                                                    gbs=gbs, num_layers=num_layers,
                                                    variance=args.min_group_scale_variance,
                                                    max_permute_len=args.max_permute_len,
                                                   ):

        metis_logger.debug(f'\n\ninter_stage_plan: {inter_stage_plan}')
        stage_performance = StagePerformance(model_config, profile_data, gpu_cluster, inter_stage_plan)
        rank_device_map = stage_performance.get_device_placement()

        intra_stage_plan_generator = IntraStagePlanGenerator(inter_stage_plan, stage_performance, layer_load_balancer, max_profiled_tp_degree, max_profiled_batch_size)

        while intra_stage_plan_generator.has_next:
            intra_stage_plan = intra_stage_plan_generator.next()
            try:
                cost = cost_estimator.get_cost(inter_stage_plan, intra_stage_plan.strategies,
                                               intra_stage_plan.layer_partition, rank_device_map)
                metis_logger.debug(f'cost: {cost}')
                estimate_costs.append((inter_stage_plan.node_sequence, inter_stage_plan.device_groups,
                                       intra_stage_plan.strategies, inter_stage_plan.batches,
                                       intra_stage_plan.layer_partition, intra_stage_plan.num_repartition, cost))
            except KeyError as e:
                metis_logger.critical(f'KeyError: {e}')

    return estimate_costs
