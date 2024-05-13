#!/bin/bash
# bash hf2mcore_qwen1.5_moe_convertor.sh 2.7B /mnt/qwen-ckpts/Qwen1.5-MoE-A2.7B /mnt/qwen-ckpts/Qwen1.5-MoE-A2.7B-Chat-to-mcore-tp1-pp1-ep4 1 1 4 false

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
mg2hf=$7
HF_CKPT_PATH=$8

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $(dirname $( dirname ${CURRENT_DIR})))
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATH}:${MEGATRON_PATH}/Megatron-LM-240405

if [ $MODEL_SIZE = 2.7B ]; then

HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
NUM_LAYERS=24
INTERMEDIATE_SIZE=5632
MOE_INTERMEDIATE_SIZE=1408
SHARED_EXPERT_INTERMEDIATE_SIZE=5632
EXTRA_VOCAB_SIZE=293
NUM_EXPERTS=60
EXPERTS_TOPK=4

gqa_options=""

cpu_options=" \
            --use-cpu-initialization"

fi


if [ $NUM_EXPERTS -gt 0 ]; then
    expert_options=" \
                --moe-router-topk ${EXPERTS_TOPK} \
                --num-experts ${NUM_EXPERTS} \
                --target-expert-model-parallel-size ${EP}"
fi

if [ $mg2hf = true ]; then
    convert_options=" \
                --convert-checkpoint-from-megatron-to-transformers \
                --hf-ckpt-path ${HF_CKPT_PATH}"

elif [ $mg2hf = false ]; then
    convert_options=""
fi


DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

torchrun ${DISTRIBUTED_ARGS} hf2mcore_qwen1.5_moe.py \
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
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --shared-moe-ffn-hidden-size ${SHARED_EXPERT_INTERMEDIATE_SIZE} \
    --ffn-hidden-size ${INTERMEDIATE_SIZE} \
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
    ${expert_options} \
    ${convert_options} \
    ${gqa_options} \
    ${cpu_options}


ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"