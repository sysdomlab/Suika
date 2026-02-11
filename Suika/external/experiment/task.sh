# !/bin/bash
source /opt/conda/bin/activate base
source ~/.bashrc

## UNCOMMENT TO DISABLE IB FOR CLUSTER-B TEST
# export NCCL_IB_DISABLE=1

# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_NCCL_TRACE_BUFFER_SIZE=10
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export NCCL_NVLS_ENABLE=0
export NCCL_SHM_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export NCCL_SOCKET_IFNAME="eth0"
export NCCL_IB_HCA="^mlx5_4,mlx5_bond_0"
export CUDA_VISIBLE_DEVICES=""
env | grep NCCL
env | grep CUDA
# python ./task_profile.py $1 $2
python ./task.py $1 $2
