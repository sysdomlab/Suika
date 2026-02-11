# !/bin/bash
# source /opt/conda/bin/activate base
# source ~/.bashrc

## Please adjust CUDA/NCCL args to comply with your cluster's configuration.

# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_NCCL_TRACE_BUFFER_SIZE=10
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_NVLS_ENABLE=0
# export NCCL_ALGO=NVLS,RING
export NCCL_SHM_DISABLE=1
export NCCL_CUMEM_ENABLE=0
# export NCCL_SOCKET_IFNAME="eth0"
export CUDA_VISIBLE_DEVICES=""
# export NCCL_IB_HCA="^mlx5_4,mlx5_bond_0"

## UNCOMMENT FOR HALF IB NICS (CLUSTER-B)
# export NCCL_IB_HCA="=mlx5_0,mlx5_2,mlx5_5,mlx5_7"

## UNCOMMENT FOR NO IB NICS (CLUSTER-C)
# export NCCL_IB_DISABLE=1

## Debug: print relevant env vars to verify settings
env | grep NCCL
env | grep CUDA

# python ./task_profile_metis.py $1 $2
# python ./task_profile.py $1 $2

## Entry point
python ./task.py $1 $2
