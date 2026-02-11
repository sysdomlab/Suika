# Helper scripts to test the correctness of scripts. Not for physical experiments.

from Suika import *
import Suika.cluster.scheduler as sch_module
from Suika.external.experiment.trace import Trace
from Suika.utils import get_logger
from Suika.cluster.policy import policy_base_impl
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
    
    # log configuration
    print(f"""
RUN [{args.casename}]: {args.policy:6} || {args.config} || {args.trace} || {args.intensity} || base-{policy_base_impl}
          """
          )
    
    # check file existence
    import os
    assert os.path.exists(args.trace), f"Trace file {args.trace} does not exist"
    assert os.path.exists(args.config), f"Config file {args.config} does not exist"

    # js = trace.run(sch) # pass
    target = f"./results_back/clusterA/{args.policy}-clusterA.pth"
    if not os.path.exists(target):
        print(f"Target file {target} does not exist, skipping creating dummy .pth results")
        return
    else:
        js = torch.load(
            target,
            weights_only=False,
        )

        retry = 3
        while retry > 0:
            try:
                os.makedirs("./results", exist_ok=True)
                torch.save(js, f"./results/{args.policy}-{args.casename}.pth")
                break
            except Exception as e:
                print(f"fail to save job statistics {e}")
                # raise e
                retry -=1

if __name__ == "__main__":
    main()