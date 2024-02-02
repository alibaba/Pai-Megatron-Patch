#!/bin/bash
# sh run_evaluate_huggingface_qwen.sh dsw ../../../PAI-Megatron-Patch 1.8B 1 2048 128 0 fp16 /mnt/qwen-datasets/alpaca_zh-qwen-train.json /mnt/qwen-ckpts/Qwen-1_8B
# sh run_evaluate_huggingface_qwen.sh dsw ../../../PAI-Megatron-Patch 7B 1 2048 128 0 fp16 /mnt/qwen-datasets/alpaca_zh-qwen-train.json /mnt/qwen-ckpts/Qwen-7B
# sh run_evaluate_huggingface_qwen.sh dsw ../../../PAI-Megatron-Patch 14B 1 2048 128 0 fp16 /mnt/qwen-datasets/alpaca_zh-qwen-train.json /mnt/qwen-ckpts/Qwen-14B

set -e
ENV=$1
MEGATRON_PATCH_PATH=$2
MEGATRON_PATH=${MEGATRON_PATCH_PATH}/Megatron-LM-231007
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
if [ $ENV = dsw ]; then
export CUDA_VISIBLE_DEVICES=0
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1

elif [ $ENV = dlc ]; then

NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}

fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

MODEL_SIZE=$3
BATCH_SIZE=$4
SEQ_LEN=$5
PAD_LEN=$6
EXTRA_VOCAB_SIZE=$7
PR=$8
DATASET_PATH=$9
PRETRAIN_CHECKPOINT_PATH=${10}


if [ $MODEL_SIZE = 1.8B ]; then

NUM_LAYERS=24
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
INTERMEDIATE_SIZE=5504

elif [ $MODEL_SIZE = 7B ]; then

NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
INTERMEDIATE_SIZE=11008

elif [ $MODEL_SIZE = 14B ]; then

NUM_LAYERS=40
HIDDEN_SIZE=5120
NUM_ATTN_HEADS=40
INTERMEDIATE_SIZE=13696
NUM_HEAD_KV=40

fi

if [ $PR = fp16 ]; then
    pr_options=" \
            --fp16"
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
fi

if [ $PRETRAIN_CHECKPOINT_PATH != none ]; then
    load_options=" \
            --load $PRETRAIN_CHECKPOINT_PATH"
fi


megatron_options=" \
        --transformer-type huggingface \
        --valid-data-path ${DATASET_PATH}
        --micro-batch-size ${BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --log-interval 1 \
        --eval-interval 100 \
        --eval-iters 10 \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --no-load-optim \
        --num-workers 0 \
        --dataset LLama-SFT \
        --use-distributed-optimizer \
        --max-padding-length ${PAD_LEN} \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --patch-tokenizer-type QwenTokenizer
        "

run_cmd="torchrun $DISTRIBUTED_ARGS evaluate_huggingface_qwen.py
 ${megatron_options} ${pr_options} ${load_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
