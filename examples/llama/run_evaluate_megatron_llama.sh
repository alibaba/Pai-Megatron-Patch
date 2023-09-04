#!/bin/bash
# sh run_evaluate_megatron_llama.sh dsw /workspace/Megatron-LM /workspace/PAI-Megatron-Patch/ 7B 1 2048 80 1 fp16 1 1 /mnt/llama-datasets/alpaca_data.json /mnt/llama-ckpts/llama-7b-hf-to-megatron-tp1-pp1
# sh run_evaluate_megatron_llama.sh dsw /workspace/Megatron-LM /workspace/PAI-Megatron-Patch/ 13B 1 2048 80 16 fp16 1 1 /mnt/llama-datasets/wudao_train.jsonl /mnt/llama-ckpts/Ziya-LLaMA-13B-to-megatron-tp1-pp1

set -e
ENV=$1
MEGATRON_PATH=$2
MEGATRON_PATCH_PATH=$3
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
if [ $ENV = dsw ]; then
export CUDA_VISIBLE_DEVICES=0,1
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
PAD_LEN=$7
EXTRA_VOCAB_SIZE=$8
PR=$9
TP=${10}
PP=${11}
DATASET_PATH=${12}
PRETRAIN_CHECKPOINT_PATH=${13}


if [ $MODEL_SIZE = 7B ]; then

NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
INTERMEDIATE_SIZE=11008

elif [ $MODEL_SIZE = 13B ]; then

NUM_LAYERS=40
HIDDEN_SIZE=5120
NUM_ATTN_HEADS=40
INTERMEDIATE_SIZE=13824

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
        --max-position-embeddings ${SEQ_LEN} \
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
        --max-padding-length ${PAD_LEN} \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --swiglu \
        --position-embedding-type rotary \
        --untie-embeddings-and-output-weights \
        --patch-tokenizer-type LLamaTokenizer \
        --recompute-activations \
        --sequence-parallel
        "

run_cmd="torchrun $DISTRIBUTED_ARGS evaluate_megatron_llama.py
 ${megatron_options} ${pr_options} ${load_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
