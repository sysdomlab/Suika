from Suika import *
import torch

target_phy_path = "results/clusterA/{policy}-clusterA.pth"
target_simu_path = "results/simu-main-e2e/{policy}-simu-main-e2e.pth"
policies = ["suika", "sia", "rubick", "fcfs"]

def load_and_compare(policy):
    phy_js = torch.load(target_phy_path.format(policy=policy), weights_only=False)
    simu_js = torch.load(target_simu_path.format(policy=policy), weights_only=False)

    phy_ajs = analyze_job_statistics(phy_js)
    phy_jct = phy_ajs["avg_jct"]
    simu_ajs = analyze_job_statistics(simu_js)
    simu_jct = simu_ajs["avg_jct"]
    
    error = abs(phy_jct - simu_jct) / simu_jct * 100

    print(f"Policy: {policy.upper():6} || phy JCT: {phy_jct:.2f}, simu JCT: {simu_jct:.2f} || Error: {error:.2f}%")

if __name__ == "__main__":
    print("""
--=================================================================--
  Comparing Physical and Simulation Results for Main E2E Experiment
--=================================================================--
          """)
    for policy in policies:
        load_and_compare(policy)