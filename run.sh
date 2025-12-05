#docker loading
docker run \
    --name qiang_vllm-projectv0.9.1-blockwise-int8-rocm643-v0_final \
    -itd \
    --network=host \
    --group-add=video \
    --ipc=host \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device /dev/kfd \
    --device /dev/dri \
    --ulimit memlock=-1:-1 \
    --shm-size=128G \
    --device /dev/infiniband \
    -v /boot:/boot \
    -v /home:/home \
    mkmhub.amd.com/sw-cserocmopt-dev/vllm-rocm:vllm-projectv0.9.1-blockwise-int8-rocm643-v0_final bash

#network settings
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3   #RoCEv2
export NCCL_DEBUG=INFO

export GLOO_SOCKET_IFNAME=enp196s0np0 # network interface
export NCCL_SOCKET_IFNAME=enp196s0np0

export NCCL_IB_HCA=mlx5_0  # notice
export NCCL_NET_GDR_LEVEL=5     # high perf mode
export NCCL_NET_GDR_READ=1      # GPU direct RDMA read

# 通道和缓冲区
export NCCL_MIN_NCHANNELS=4
export NCCL_MAX_NCHANNELS=4

# 网络可靠性
export NCCL_IB_TIMEOUT=22        # 避免短时拥塞误判
export NCCL_IB_RETRY_CNT=7       # 增加重试次数

#Ray settings
apt install iproute2
ray start --head \
  --node-ip-address=192.168.23.8 \
  --port=6379 > ray.log 2>&1 

ray start --address='192.168.23.8:6379' 
ray status
ray list nodes --detail  

export VLLM_HOST_IP=192.168.23.8 # cse-ai-8

cd /opt/rocm 
tar -xvf rccl_deps.tar --strip-components=1

export LD_LIBRARY_PATH=/root/rccl_deps/lib:$LD_LIBRARY_PATH
#vLLM serve launch
VLLM_TORCH_PROFILER_DIR=/vllm_v0.9.1/ \
NCCL_DEBUG=INFO \
VLLM_PROFILER_MAX_ITERS=2 \
VLLM_TORCH_PROFILER_WITH_STACK=0 \
VLLM_TORCH_PROFILER_RECORD_SHAPES=0 \
VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_MLA_DISABLE=1 \
VLLM_USE_TRITON_FLASH_ATTN=1 \
vllm serve /home/public/model2/DeepSeek-V3-0324-BF16-Cast-To-Blockwise-Int8/ \
    --block-size 16 \
    --max-num-seqs  1 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --trust-remote-code \
    -tp 8 \
    -pp 4 \
    --distributed-executor-backend ray \
    --seed 0 \
    --generation-config auto \
    --override_generation_config '{"temperature": 0}' > serve_log.txt 2>&1 

CONCURRENT=4
#benchmark test
python /vllm_v0.9.1/benchmarks/benchmark_serving.py \
    --backend vllm \
    --model /home/public/model2/DeepSeek-V3-0324-BF16-Cast-To-Blockwise-Int8/ \
    --dataset-name random \
    --num-prompts ${CONCURRENT} \
    --seed 0 \
    --max-concurrency ${CONCURRENT} \
    --random-input-len 129 \
    --random-output-len 1024 \
    --ignore-eos > trace.log 2>&1 
    --profile
