from Suika import *
from Suika.external.Sia.goodput import fit_perf_params, GoodputFunction

def fit(model_spec):
    path = f"./profile/{model_spec}_profile.csv"
    
    ms = ModelStrategy.from_spec(model_spec)
    (target_pp, _, target_tp) = ms.pp_dp_tp

    print(f"\n<pp {target_pp}, tp {target_tp}>")
    with open(path, "r") as f:
        lines = f.readlines()
        data_entries = []
        for line in lines:
            (pp, dp, tp, gbs, n_mb), accum, optim, iter_time, _, _ = eval(line)
            n_node = (dp * tp + 8 - 1) // 8 * 1.0
            n_rep = dp * 1.0
            atomic_bsz = gbs // n_mb // n_rep * 1.0
            accum_steps = n_mb - 1
            if accum < 0:
                continue
            if tp != target_tp or pp != target_pp:
                continue
            data_entries.append([n_node, n_rep, accum_steps, atomic_bsz, accum, optim, iter_time])

    import numpy as np

    n_node, n_rep, accum_steps, atomic_bsz, accum, optim, iter_time = map(lambda x: np.array(list(x)), zip(*data_entries))
    # print(n_node, n_rep, atomic_bsz, accum, optim, iter_time)
    perf_params = fit_perf_params(n_node, n_rep, atomic_bsz, accum, optim)
    print(f"model_spec: {model_spec}")
    asdict = {k: float(v) for k, v in perf_params._asdict().items()}
    print(f"\n{asdict}\n")
    goodput_fn = GoodputFunction(perf_params)
    avg_error = 0
    for n_node_, n_rep_, accum_steps_, atomic_bsz_, iter_time in zip(n_node, n_rep, accum_steps, atomic_bsz, iter_time):
        bsz = n_rep_ * atomic_bsz_ * (accum_steps_ + 1)
        est_iter_time = 1 / goodput_fn.evaluate(accum_steps=accum_steps_, num_nodes=n_node_, num_replicas=n_rep_, atomic_bsz=atomic_bsz_) * bsz
        print(f"[{n_node_}, {n_rep_}] std: {iter_time:.2f} / est: {est_iter_time:.2f} / error {abs(est_iter_time - iter_time) / iter_time * 100:.1f}")
        avg_error += abs(est_iter_time - iter_time) / iter_time * 100
    avg_error /= len(n_node)
    print(f"avg_error: {avg_error:.1f}")

model_specs = [
    "GPT-350M", "GPT-1.3B", "GPT-2.6B", "GPT-6.7B", "GPT-15B", "Llama2-7B", "Llama2-13B", 
    "Qwen2-0.5B", "Qwen2-1.5B", "Qwen2-7B",
    "Llama-30B",
    "GPT-39B",
    "GPT-76B", "GPT-175B", "Llama-65B", "Llama2-70B", "Qwen2-72B",
]

for model_spec in model_specs:
    fit(model_spec)