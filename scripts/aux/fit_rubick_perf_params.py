from Suika import *
from Suika.cluster.estimator import Estimator
from Suika.external.Rubick.pred_throughput import fit_perf_params, ThroughputFunction

def fit(model_spec):
    path = f"./profile/{model_spec}_profile.csv"
    # path = f"./{model_spec}_profile.csv"
    ms = ModelStrategy.from_spec(model_spec)
    (_, _, target_tp) = ms.pp_dp_tp

    with open(path, "r") as f:
        lines = f.readlines()
        data_entries = []
        for line in lines:
            (pp, dp, tp, gbs, n_mb), accum, optim, iter_time, _, _ = eval(line)
            ms = ModelStrategy.from_spec(model_spec)
            ms.pp_dp_tp = (pp, dp, tp)
            ms.global_batch_size = gbs
            ms.num_micro_batch = n_mb
            if not ms.validate():
                continue
            data_entries.append([ms, iter_time])
            
    import numpy as np

    mss, iter_times = map(lambda x: list(x), zip(*data_entries))
    # print(n_node, n_rep, atomic_bsz, accum, optim, iter_time)
    perf_params = fit_perf_params(mss, iter_times)
    print(f"model_spec: {model_spec}")
    asdict = {k: float(v) for k, v in perf_params._asdict().items()}
    print(f"\n{asdict}\n")
    mc = ModelConfig.from_spec(model_spec)
    goodput_fn = ThroughputFunction(mc, perf_params)
    avg_rubick_error = 0
    avg_suika_error = 0
    for ms, iter_time in zip(mss, iter_times):
        est_iter_time = goodput_fn.predict_throughput_from_ModelStrategy(ms, True)
        pc = ms.get_plan_config()
        C =  ms.model_config.estimator_C
        C["optimizer"] = 0.0
        est = Estimator(get_default_rm(), C)
        suika_est_iter_time = est.estimate(pc)
        rubick_error = abs(est_iter_time - iter_time) / iter_time * 100
        suika_error = abs(suika_est_iter_time - iter_time) / iter_time * 100
        print(f"{ms.pp_dp_tp} - std: {iter_time:.2f} / est: {est_iter_time:.2f} / error {rubick_error:.1f}% / suika est {suika_est_iter_time:.2f} / error {suika_error:.1f}%")
        avg_rubick_error += rubick_error
        avg_suika_error += suika_error
    avg_rubick_error /= len(mss)
    avg_suika_error /= len(mss)
    print(f"avg_error: {avg_rubick_error:.1f}% / {avg_suika_error:.1f}%")

model_specs = [
    "GPT-350M", "GPT-1.3B", "GPT-2.6B", "GPT-6.7B", "GPT-15B", "Llama2-7B", "Llama2-13B", "Qwen2-0.5B", "Qwen2-1.5B", "Qwen2-7B",
    "Llama-30B",
    "GPT-39B"
    "GPT-76B", "GPT-175B", "Llama-65B", 
    "Llama2-70B", 
    "Qwen2-72B",
]

for model_spec in model_specs:
    fit(model_spec)