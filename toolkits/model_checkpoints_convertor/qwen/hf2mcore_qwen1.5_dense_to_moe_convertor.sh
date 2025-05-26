#!/bin/bash
# bash hf2mcore_qwen1.5_dense_to_moe_convertor.sh 1.8B /mnt/qwen-ckpts/Qwen1.5-1.8B /mnt/qwen-ckpts/Qwen1.5-MoE-A2.7B-to-mcore-tp1-pp1-ep4 1 1 4 60 4 1408

set -e
export CUDA_VISIBLE_DEVICES=7
START_TIME=$SECONDS
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

MODEL_SIZE=$1
SOURCE_CKPT_PATH=$2
TARGET_CKPT_PATH=$3
TP=$4
PP=$5
EP=$6
NUM_EXPERTS=$7
NUM_SPLITS=$8
MOE_INTERMEDIATE_SIZE=$9


CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATCH_PATH=$( dirname $(dirname $( dirname ${CURRENT_DIR})))
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATCH_PATH}:${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-240405

if [ $MODEL_SIZE = 1.8B ]; then

NUM_LAYERS=24
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
INTERMEDIATE_SIZE=5504
EXTRA_VOCAB_SIZE=293
SHARED_EXPERT_INTERMEDIATE_SIZE=$(( ${MOE_INTERMEDIATE_SIZE} * 4 ))

gqa_options=""
cpu_options=" \
            --use-cpu-initialization"

fi

DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

if [ $MODEL_SIZE != 32B ]; then

torchrun ${DISTRIBUTED_ARGS} hf2mcore_qwen1.5_dense_mha_to_moe.py \
    --load ${SOURCE_CKPT_PATH} \
    --save ${TARGET_CKPT_PATH} \
    --target-tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --micro-batch-size 1 \
    --save-interval 1 \
    --fp16 \
    --swiglu \
    --norm-epsilon 1e-6 \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --ffn-hidden-size ${INTERMEDIATE_SIZE} \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --shared-moe-ffn-hidden-size ${SHARED_EXPERT_INTERMEDIATE_SIZE} \
    --num-attention-heads ${NUM_ATTN_HEADS} \
    --max-position-embeddings 1 \
    --seq-length 1 \
    --no-async-tensor-model-parallel-allreduce \
    --patch-tokenizer-type Qwen2Tokenizer \
    --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
    --untie-embeddings-and-output-weights \
    --no-rope-fusion \
    --use-rotary-position-embeddings \
    --transformer-impl transformer_engine \
    --disable-bias-linear \
    --normalization RMSNorm \
    --add-qkv-bias \
    --use-mcore-models \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --enable-shared-expert \
    --num-experts ${NUM_EXPERTS} \
    --num-splits ${NUM_SPLITS} \
    --target-expert-model-parallel-size ${EP} \
    ${gqa_options} \
    ${cpu_options}

fi

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"