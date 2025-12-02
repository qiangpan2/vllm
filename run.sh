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
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=bnxt_re
export GLOO_SOCKET_IFNAME=enp196s0np0 # network interface
export NCCL_SOCKET_IFNAME=enp196s0np0

#Ray settings
apt install iproute2
# ray start --head --node-ip-address=192.168.23.8 --port=6379 --num-gpus=8
# ray start --address='192.168.23.8:6379' --num-gpus=8
ray status

export VLLM_HOST_IP=192.168.23.8 # cse-ai-8

#vLLM serve launch
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
    --override_generation_config '{"temperature": 0}'

#benchmark test
python /vllm_v0.9.1/benchmarks/benchmark_serving.py \
    --backend vllm \
    --model /home/public/model2/DeepSeek-V3-0324-BF16-Cast-To-Blockwise-Int8/ \
    --dataset-name random \
    --num-prompts 1 \
    --seed 0 \
    --max-concurrency 1 \
    --random-input-len 16 \
    --random-output-len 16 \
    --ignore-eos
