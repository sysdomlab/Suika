# This submodule is adapted from 'Metis: Fast Automatic Distributed Training on Heterogeneous GPUs' open artifacts

Major adaptations to fit Metis into our framework are marked with `# @adaptation` at head of each file.

Workflow:
1. we provide Metis with resources information from ResourceManager
2. run Metis optimization
3. convert Metis plan descriptor to PlanConfig 
4. execute with our runtime.