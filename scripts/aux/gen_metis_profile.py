from Suika import *
from Suika.utils import get_logger
from Suika.cluster.resource import default_acc_type
from Suika.cluster.job import JobStatus
from Suika.external.experiment.trace import TraceEntry

def get_max_tp_max_bs(model_spec: str):
    pc = get_plan_config_from_spec(model_spec)
    max_tp = pc.max_tp_allowed
    gmbs = pc.global_micro_batch_size()
    for max_bs_exp in range(10):
        if 2 ** (max_bs_exp + 1) > gmbs:
            break
    max_bs = 2 ** max_bs_exp
    print(f"[{model_spec} - max_tp {max_tp} max_bs {max_bs} | gmbs {gmbs}]")
    return max_tp, max_bs

def profile_file_template(n_layer, layer_compute, optimizer_time, batch_gen_time, layer_mem, layer_param_in_bytes):
  # all layers are the same and repetive.
  layer_compute_total_ms = [layer_compute] * n_layer
  forward_backward_time_ms = sum(layer_compute_total_ms) + 0.1

  return {
    "model": {
      "parameters": {
        "parameters_per_layer_bytes": [layer_param_in_bytes] * n_layer
      }
    },
    "execution_time": {
      "optimizer_time_ms": optimizer_time,
      "layer_compute_total_ms": layer_compute_total_ms,
      "batch_generator_time_ms": batch_gen_time,
      "forward_backward_time_ms": forward_backward_time_ms
    },
    "execution_memory": {
      "layer_memory_total_mb": [layer_mem] * n_layer
    }
  }

import os
import time
def get_target_file_name(device, tp, bs):
    return f"DeviceType.{device}_tp{tp}_bs{bs}.json"
    
def power_of_2_list(top):
    return [2 ** i for i in range(10) if 2 ** i <= top]

def post_process(model_spec, max_tp, max_bs):
    target = f"./Suika/external/Metis/profile_data/{model_spec}"
    if not os.path.exists(target):
        os.mkdir(target)
    pc = get_plan_config_from_spec(model_spec)

    with open(f"tasks/{model_spec}_profile_metis.csv") as f:
        lines = f.readlines()

    raw_data = {}
    for line in lines:
        device = default_acc_type
        (tp, bs), compute_time, optimizer_time, batch_gen_time, max_memory, param = eval(line)
        compute_time *= 1000
        optimizer_time *= 1000
        batch_gen_time *= 1000
        raw_data[(tp, bs)] = [
            compute_time, optimizer_time, batch_gen_time, max_memory, param
        ]

    for tp in power_of_2_list(max_tp):
        for bs in range(1, max_bs + 1):
            sum_compute_time, sum_batch_gen_time, sum_max_memory = 0, 0, 0
            optimizer_time, param = 0, 0
            for sub_bs in power_of_2_list(max_bs):
                if sub_bs & bs:
                    compute_time, optimizer_time, batch_gen_time, max_memory, param = raw_data[(tp, sub_bs)]
                    sum_compute_time += compute_time
                    sum_batch_gen_time += sum_batch_gen_time
                    sum_max_memory += max_memory
            compute_time, batch_gen_time, max_memory = sum_compute_time, sum_batch_gen_time, sum_max_memory

            file = target + "/" + get_target_file_name(device, tp, bs)

            with open(file, "w") as f:
                import json
                json.dump(profile_file_template(
                            n_layer=pc.total_num_layer(), 
                            layer_compute=compute_time,
                            optimizer_time=optimizer_time,
                            batch_gen_time=batch_gen_time,
                            layer_mem=max_memory,
                            layer_param_in_bytes=param,
                            ), 
                        f,
                        indent=2
                        )


def profile_one_case(model_spec, tp, bs, scheduler: Scheduler):
    pc = get_plan_config_from_spec(model_spec, n_layer=1, pp_dp_tp=(1, 1, tp), gbs=bs, n_mb=1)
    print(f"> {model_spec}x{tp}x{bs} start @ {scheduler._record_time():.1f}")
    # print(f"> pc {pc}")
    job = TraceEntry.from_csv((model_spec, 0, 50), "./checkpoint").job
    job.usr_plan_config = pc

    scheduler.submit(job)
    while scheduler.active_jobs:
        time.sleep(1)
    if job.status != JobStatus.COMPLETED:
        print(f"failed to run {model_spec}x{tp}x{bs}")
        exit(1)
    print(f"< {model_spec}x{tp}x{bs} done @ {scheduler._record_time():.1f}")


def profile_one_model(model_spec, scheduler):
    max_tp, max_bs = get_max_tp_max_bs(model_spec)
    # shortcut
    print(f">> {model_spec} start @ {scheduler._record_time():.1f}")
    for tp in power_of_2_list(max_tp):
        for bs in power_of_2_list(max_bs):
            profile_one_case(model_spec, tp, bs, scheduler)
            time.sleep(1)
    post_process(model_spec, max_tp, max_bs)
    print(f"<< {model_spec} done @ {scheduler._record_time():.1f}")

model_lists = [
    "GPT-350M", "GPT-1.3B", "GPT-2.6B", "GPT-6.7B", 
    "GPT-15B", 
    "Llama2-7B", 
    "Llama2-13B", 
    "Qwen2-0.5B", 
    "Qwen2-1.5B", "Qwen2-7B",
    "GPT-39B", "Llama-30B"
]

if __name__ == "__main__":
    import argparse
    import logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ALL")
    model: str = parser.parse_args().model


    scheduler = GeneralScheduler("./suika.json", "fcfs", ff_enable=False, logger=get_logger("profile_metis", console_level=logging.ERROR))

    if model.upper() == "ALL":
        for model in model_lists:
            profile_one_model(model, scheduler)
    else:
        profile_one_model(model, scheduler)