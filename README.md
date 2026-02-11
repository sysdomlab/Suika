# Artifact for Suika (EuroSys'26)

---

# Overview

This repository hosts the artifact for the EuroSys'26 paper *Suika: Efficient and High-quality Re-scheduling of 3D-parallelized LLM Training Jobs in Shared Clusters*.

The README document is organized as follows:
- Getting started
- Reproducing main results
  - Simulation guide
  - Physical testbed guide
- Repository structure walk-through
- License


---

# Get Started

## Testbed

**For physical testbed**: Suika does not rely on exclusive hardware or OS features beyond those commonly available on NVIDIA GPUs and Linux distributions. To reproduce similar results as presented in the paper, high-end hardware, as prevalent in LLM training, is necessary.

The table summarizes our physical testbed setup:
- GPU: NVIDIA H100 SXM (80G, 700W), x8 GPUs per node.
- CPU: Intel CPU, 128 cores
- Host memory: 1024 GB
- OS: Ubuntu 20.04
- Connection:
  - Intra-node: NVLink with NVSwitch (900 GB/s).
  - Inter-node: InfiniBand NDR (400 Gbps), x8 NICs per node.
- Nodes: 8 nodes and 64 GPUs in total.

**For simulation testbed**: A typical Linux laptop will do. We suggest using a multi-core CPU with >4 GHz core frequency for faster completion.

---

## Dependencies 

### Use Docker image:

The docker file is available at [Dockerfile](Dockerfile). Build with command:

```
# Go to repository root
docker build -t suika:cu126 .

docker run --name suika_runtime -d \
  --gpus all \
  --network host \
  -w /root/workspace \
  --shm-size=4g \
  suika:cu126 \
  tail -f /dev/null

docker exec -it suika_runtime bash
```

The docker is capable for both physcial and simulated experiments.
However, some hardware-specific adjustments are still necessary (see instructions in Reproduce Main Results section).

You may change the base image for a different CUDA version, but torch version should be kept consistent.

---

### To install from scratch

**Step 1**. Install Python 3.11, e.g., with conda:
```bash
conda create -n Suika python=3.11
conda activate Suika
```

**Step 2**. Install Python dependencies:
```bash
pip install -r requirements.txt
```

**Step 3**. Install the Suika module:

```bash
pip install -e .
```

**Step 4**. Install the C++ backend solver (ensure a C++ compiler is installed and pybind11 headers are in the include path as needed):
```bash
cd Suika/cluster/cpp_backend && python setup.py build_ext --inplace
```

---

# Reproduce Main Results
> For AE reviewers
>
> Suika can be deployed and is validated on a physical testbed. However, since physical experiments are time-consuming and costly, it is hard to provide a testing platform for AE reviewers.
> - Reproducing the entire physical experiment would require exclusive access to 64 H100 GPUs for more than a week.
> - Even reproducing only the main end-to-end results would take over 12 hours.
> 
> To promote validation of physical results, we collect key data (including job submission, reconfiguration and completion timestamps, elastic plans and their throughputs, etc.) that are sufficient to replay the experiment. We use these data for further analysis and visualization in the paper. These data are stored under `./results` as `.pth` files.
>
> Regardless, a complete guide to deploy Suika for physical experiments is provided. Validation of physical functionality can be conducted on a single-node setup (see below).

## General Preparation

**Cluster configuration file**:
Suika describes the cluster with JSON. The meaning of each field is straightforward, as exemplified in `cluster64.json`:
```json
{
  "accelerator": {
    "H100": {
      "TFLOPS": 989.4, # FP16
      "memory": 81920 # in MB
    }
  },
  "node": {
     "10.244.0.97": { # worker IP
      "node_id": 0,
      "acc_type": "H100",
      "acc_num": 8
    },
    ...
    "10.244.7.89": {
      "node_id": 7,
      "acc_type": "H100",
      "acc_num": 8
    }
  },
  "network": {
    "H100": [300000, 2048], # effective bandwidth, saturation packet-size
    "H100,H100": 50000, # in MB/s
  },
  "policy": "fcfs", # default policy, can be and usually is overridden later
  "scheduler_addr": "10.244.4.33", # IP address of scheduler node
  "job_port_begin": 40402, # scheduler port range
  "worker_port_begin": 12355 # worker port range
}
```

Cluster files are under `./configs`. Their names indicate the number of GPUs in the cluster.

For the physical testbed, the `scheduler_addr` needs to be adapted to your scheduler node's real IP address.

**Trace file**:

Suika describes a job (trace entry) with a tuple of (application, submit_time, num_iterations).
```csv
Llama2-13B,0.0,3998
Llama2-7B,75.0,6166
GPT-6.7B,825.0,16
...
```

Application types are defined in `Suika/models`, with pre-profiled performance model parameters for baselines. Table 1 in the paper provides an application overview.

Trace files are also under `./configs`. Their names indicate the experiments they correspond to.

---

## Physical Experiment

### Basic Method

> This section describes the fundamentals to run the physical experiment. High-level combined scripts are introduced later.

**Step 0**: Properly complete the general preparation above. 

Additionally:
- To run baselines requiring checkpointing, a high-speed NFS is required. Place the workspace directory on the NFS (the checkpoint path is by default under the workspace).
- **Adjust env vars in `tasks/task.sh` to fit your cluster**. Some arguments (e.g., ethernet interface name) are hardware-specific, and thus need adjustment.

> Debugging physical deployment is tricky due to vast variants of hardware environment. We are happy to provide assistance and please contact us if you encounter troubles.

> Regarding LOGGING: the majority of logs go to `./log/`, and worker logs go to `./tasks/log`


**Step 1**: Launch the executor on each worker node
```bash
python -m Suika.cluster.executor --node-id ${NODE_ID} --config ${CLUSTER_CONFIG}
```
where `CLUSTER_CONFIG` is the path to the cluster configuration JSON, and `NODE_ID` is a per-node unique identifier that must exist in the cluster configuration JSON.

> It's suggested to run this step with tools like Slurm or an SSH remote execution script.

**Step 2**: Launch the scheduler and submit the trace on the scheduler node.
```bash
python scripts/phy/run_scheduler.py --policy ${policy} --trace ${trace} --ckpt checkpoint_${policy} --casename ${case} > ${policy}_${case}.log
```

Note that the scheduler waits for executor connections from all expected worker nodes before serving the workload. There is a timeout mechanism.

### Combined Scripts


**Please configure SSH** so that the scheduler node can SSH onto worker node $i$ with the SSH alias `H100-worker-0{i}` without a password. Then:

For main end-to-end experiment:

```bash
bash scripts/phy/main_e2e.sh
```

For sensitivity study a–f:

```bash
bash scripts/phy/sensitivity_study_{a~f}.sh
```

The underlying entry-point script is `scripts/phy/aux_run.sh`, and there are other auxiliary scripts with `aux_` prefixes.

The results (data collected to replay the experiment) are saved as `.pth` files under `./results`. Finally, run
```bash
python scripts/phy/organize_result.py
```
to group these files into subfolders for each case, as expected by visualizing scripts.

---

## Minimal Physical Functionality Validation

On a x8 GPU(80GB) node (serving as both scheduler and worker node), run
```bash
bash scripts/phy/minimal_test.sh
```

It is possible to run on a single GPU (80 GB) node. Adjust "acc_num" to 1 in `configs/cluster_minimal.json:line 12`.
To run with smaller GPU memory, adjust "global_batch_size" in `Suika/models/gpt_model.py:line 413`.

---

## Simulation

The entry point script is
```bash
python run_simulation.py --policy ${POLICY} --trace ${TRACE} --config ${CLUSTER_CONFIG} --casename ${CASE}
```

To replay main end-to-end experiment with simulation and validate fidelity,
```bash
bash scripts/simu/simu_main_e2e.sh
```

To run large-scale simulation on 1024 GPUs,
```bash
bash scripts/simu/simu_large_scale.sh
```

To record schedule latency,
```bash
bash scripts/simu/simu_schedule_overhead.sh
```

Please note that these scripts in general do not visualize the results. Use jupyter notebook in the later section for visualization.
Please note that these scripts in general do not visualize results. Use the Jupyter notebooks in the later section for visualization.

## Auxiliary

Auxiliary scripts are stored under `scripts/aux`, for baseline profiling and other minor functionality. Please follow each baseline's respective guide for details.

---


## Visualization


### Cached Results

We provide all our experiments' raw results under `./results`, and visualization notebooks and printed PDFs under `./viz`.

To visualize your data, please comply with existing naming rules and folder structure.


### Figure 11. End-to-end performance with Cluster-A.

Notebook: `visualize_main_e2e.ipynb`. 

Data required: cluster-A (main e2e)

### Figure 12. Comparisons on performance modeling.

Notebook: `visualize_performance_model.ipynb`. 

Data required: cluster-A (main e2e)

### Figure 13. (a)  Plan quality of LlaMA2-13B as resource expands

Notebook: `visualize_decision_quality.ipynb`

Data required: micro_scopic.csv

### Figure 13. (b)  Impact on individual jobs relative to the non-elastic scheduler

Notebook: `visualize_job_acceleration_cdf.ipynb`

Data required: cluster-A (main e2e)

### Figure 14. Comparisons on redeployment cost.

Notebook: `visualize_redeployment_time.ipynb`

Data required: cluster-A (main e2e)

### Figure 15. Sensitivity analysis.

Notebook: `visualize_sensitivity_study.ipynb`

Data required: 
- (a): clusterA (main e2e), clusterB (half IB), clusterC (no IB), 
- (b): int0.75, clusterA (int1.00), int1.25, int1.5, int2.0
- (c): lj0.5, clusterA (lj1.0), lj1.5
- (d): th-exp-ablation
- (e): sjf-ablation
- (f): clusterA, philly2, helios_venus, helios_uranus

### Figure 16. Results in large-scale simulations. (a) End-to-end performance

Notebook: `visualize_main_simulation.ipynb`

Data required: 
- simu-trace1, simu-trace2, simu-trace3, simu-trace4

Please note that there is randomness in baseline scheduling, and therefore results may fluctuate slightly (around 1–2%).

### Figure 16. Results in large-scale simulations. (b) Scheduling latency

Notebook: `visualize_schedule_overhead.ipynb`

Data required: schedule-overhead

---

# Repo Structure Walk-Through

This repository is structured as follows:

- `configs/` — cluster configurations and trace files.
- `results/` — raw experiment results (replayable `.pth` files).
- `fig/` — figures used in the paper.
- `Suika/` — the core module and runtime.
- `tasks/` — worker-side workspace and helper scripts.
- `viz/` — visualization notebooks and generated PDFs.
- `scripts/` — experiment scripts (physical, simulation, auxiliary).

Key contents under `Suika/`:

- `cluster/` — scheduler, executor, simulator, and the C++ backend.
- `models/` — LLM performance models and utilities.
- `external/` — baseline implementations and integrations.
- `engine.py`, `pipeline_parallel.py`, `tensor_parallel.py` — SuikaEngine components.
- `migration.py` — redeployment optimization.
- `training.py` — SuikaAgent.
- `device_mesh.py`, `profiler.py`, `utils.py` — supporting infrastructure.

The code is well commented; please refer to the source code for details and feel free to contact us.

---

# License ![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.

To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/ or see the [LICENSE](LICENSE) file.