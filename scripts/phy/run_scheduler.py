from Suika import *
import Suika.cluster.scheduler as sch_module
from Suika.external.experiment.trace import Trace
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

    datetime = time.strftime("%m-%d-%H:%M:%S", time.localtime())

    schedule_interval = 60 if args.policy.lower() == "sia" else 30
    ff_time_max_range = 1e9 if args.policy.lower() == "metis" else 200
    sch_module.set_ff_time_max_range(ff_time_max_range)

    print(f"Using schedule_interval {schedule_interval} / ff_time_max_range {sch_module.default_ff_time_max_range}")
    sch = GeneralScheduler(args.config, args.policy, ff_enable=not args.ff_disable, schedule_interval=schedule_interval, logger=get_logger(f"scheduler-{datetime}-{args.policy}"), force_checkpoint_resume=args.force_ckpt)

    js = trace.run(sch)

    retry = 3
    while retry > 0:
        try:
            torch.save(js, f"result/{args.trace}-{datetime}-{args.policy}-{args.casename}.pth")
            break
        except Exception as e:
            print(f"fail to save job statistics {e}")
        retry -=1

    for k, v in analyze_job_statistics(js).items():
        sch.log(f"{k}: {v}")

    return

if __name__ == "__main__":
    main()