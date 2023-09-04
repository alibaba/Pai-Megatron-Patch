#!/bin/bash
# sh run_evaluate_megatron_starcoder.sh dsw /workspace/Megatron-LM /workspace/PAI-Megatron-Patch/ 16B 1 2048 fp16 2 1 /mnt/workspace/latest/alpaca_data.json /mnt/workspace/latest/starcoder-mg21

set -e
ENV=$1
MEGATRON_PATH=$2
MEGATRON_PATCH_PATH=$3
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
if [ $ENV = dsw ]; then
export CUDA_VISIBLE_DEVICES=0,1,2,3
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=2

elif [ $ENV = dlc ]; then

NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}

fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

MODEL_SIZE=$4
BATCH_SIZE=$5
SEQ_LEN=$6
PR=$7
TP=${8}
PP=${9}
DATASET_PATH=${10}
PRETRAIN_CHECKPOINT_PATH=${11}


if [ $MODEL_SIZE = 16B ]; then

NUM_LAYERS=40
HIDDEN_SIZE=6144
NUM_ATTN_HEADS=48
INTERMEDIATE_SIZE=24576

elif [ $MODEL_SIZE = 7B ]; then

NUM_LAYERS=42
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
INTERMEDIATE_SIZE=16384

elif [ $MODEL_SIZE = 3B ]; then

NUM_LAYERS=36
HIDDEN_SIZE=2816
NUM_ATTN_HEADS=22
INTERMEDIATE_SIZE=11264

elif [ $MODEL_SIZE = 1B ]; then

NUM_LAYERS=24
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
INTERMEDIATE_SIZE=8192

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
        --data-path ${DATASET_PATH}
        --micro-batch-size ${BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings 8192 \
        --intermediate-size ${INTERMEDIATE_SIZE} \
        --log-interval 1 \
        --eval-interval 100 \
        --eval-iters 10 \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --DDP-impl local \
        --no-load-optim \
        --no-load-rng \
        --seed 1234 \
        --num-workers 0 \
        --dataset LLama-SFT \
        --use-distributed-optimizer \
        --attention-head-type multiquery \
        --patch-tokenizer-type StarcoderTokenizerFromHF \
        --recompute-activations \
        "

run_cmd="torchrun $DISTRIBUTED_ARGS evaluate_megatron_starcoder.py
 ${megatron_options} ${pr_options} ${load_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
