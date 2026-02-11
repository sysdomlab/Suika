# LAST EDIT: 07.31
from Suika import SimulatorScheduler, analyze_job_statistics
from Suika.utils import get_logger
from Suika.cluster.resource import Accelerator
from Suika.external.experiment.trace import Trace, TraceEntry

import logging

node_per_rack=16

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a simulation with a trace file.")
    parser.add_argument("--trace", type=str, default="./trace_1024.csv", help="Path to the trace file.")
    parser.add_argument("--policy", type=str, default="suika", help="Target Policy.")
    parser.add_argument("--config", type=str, default="./suika1024.json", help="Path to the configuration file.")
    parser.add_argument("--refresh-at-event", action="store_true", help="Refresh tqdm at each event. (do not turn on for background run, as it overwhelms the log)")
    parser.add_argument("--case", type=str, default="", help="Case name")
    args = parser.parse_args()
    # set_policy_threshold_exp(1.0)
    
    trace = Trace.from_csv(args.trace)
    scheduler = SimulatorScheduler(policy=args.policy, 
                                   add_tqdm=True,
                                   logger=get_logger("simulator", console_level=logging.ERROR, level=logging.DEBUG), config=args.config, 
                                   add_tqdm_refresh_at_event=args.refresh_at_event)

    js = trace.run_simu(scheduler)
    import torch, os

        
    ajs = analyze_job_statistics(js)
    makespan = ajs["make_span"]
    gpu_time = ajs["avg_gpu_time"] * len(trace)
    n_gpu = scheduler.rm.get_total_acc_num()
    
    if "so" not in args.case:
        result_dir = f"./results/{args.case}"
        os.makedirs(result_dir, exist_ok=True)
        torch.save(js, f"{result_dir}/{args.policy}-{args.case}.pth")
        
        print(f"POLICY: {args.policy.upper()}")
        for k, v in ajs.items():
            if k != "per_job_statistics":
                print(f"{k}: {v}")
        print(f"utilization: {gpu_time / (makespan * n_gpu) * 100:.1f}%")
    else:
        print(f"POLICY: {args.policy.upper()}, done.")
    
    if "so" in args.case:
        so = scheduler.schedule_overhead
        result_dir = f"./results/schedule-overhead"
        os.makedirs(result_dir, exist_ok=True)
        torch.save(so, f"{result_dir}/{args.policy}-{n_gpu}.pth")