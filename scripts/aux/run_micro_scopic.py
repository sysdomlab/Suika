from Suika import *
import Suika.cluster.scheduler as sch_module
from Suika.external.experiment.trace import Trace, TraceEntry
from Suika.utils import get_logger
import time
import torch

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", "-t", type=str, default="./trace.csv")
    parser.add_argument("--config", "-c", type=str, default="./suika.json")
    parser.add_argument("--policy", "-p", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--ff-disable", action="store_true")
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--force-ckpt", action="store_true")
    parser.add_argument("--casename", type=str, default="")

    args = parser.parse_args()
    print(f"Running trace {args.trace} with policy {args.policy}")

    trace = Trace.from_csv(args.trace, checkpoint_dir=args.ckpt)
    for trace_entry in trace:
        trace_entry.submit_time /= args.intensity
        trace_entry.job.usr_plan_config = parse_plan_config_from_file("plan32.txt")

    datetime = time.strftime("%m-%d-%H:%M:%S", time.localtime())

    schedule_interval = 60 if args.policy.lower() == "sia" else 30
    ff_time_max_range = 1e9 if args.policy.lower() == "metis" else 200
    sch_module.set_ff_time_max_range(ff_time_max_range)

    print(f"Using schedule_interval {schedule_interval} / ff_time_max_range {sch_module.default_ff_time_max_range}")
    sch = GeneralScheduler(args.config, args.policy, ff_enable=not args.ff_disable, schedule_interval=schedule_interval, logger=get_logger(f"scheduler-{datetime}-{args.policy}"), force_checkpoint_resume=args.force_ckpt)

    js = trace.run(sch)

    with open("micro_scopic.csv", "a+") as f:
        num_acc = len(sch.rm.get_avail_acc_list())
        for k, v in js.items():
            num_reconfig = len(v["plan_config"])
            iter_time = v["iter_time"][num_reconfig - 1]
            pc = v["plan_config"][-1]
            f.write(f"========================\n")
            f.write(f"{num_acc}, {args.policy}, {iter_time}\n")
            f.write(f"{pc}\n")
            f.write(f"========================\n")

    return

if __name__ == "__main__":
    main()

"""
"""