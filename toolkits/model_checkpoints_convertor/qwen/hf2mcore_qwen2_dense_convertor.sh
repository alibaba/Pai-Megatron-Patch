#!/bin/bash
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
USE_TE=$6
MG2HF=$7
HF_CKPT_PATH=$8

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $(dirname $( dirname ${CURRENT_DIR})))
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATH}:${MEGATRON_PATH}/Megatron-LM-240612


if [ $MODEL_SIZE = 0.5B ]; then

HIDDEN_SIZE=896
INTERMEDIATE_SIZE=4864
MAX_POSITION_EMBEDDINGS=131072
MAX_WINDOW_LAYERS=24
NUM_ATTENTION_HEADS=14
NUM_HIDDEN_LAYERS=24
NUM_KEY_VALUE_HEADS=2
RMS_NORM_EPS=1e-6
ROPE_THETA=1000000
SLIDING_WINDOW=131072
EXTRA_VOCAB_SIZE=293

moe_options=" \
            "

cpu_options=""

elif [ $MODEL_SIZE = 1.5B ]; then

HIDDEN_SIZE=1536
INTERMEDIATE_SIZE=8960
MAX_POSITION_EMBEDDINGS=131072
MAX_WINDOW_LAYERS=28
NUM_ATTENTION_HEADS=12
NUM_HIDDEN_LAYERS=28
NUM_KEY_VALUE_HEADS=2
RMS_NORM_EPS=1e-6
ROPE_THETA=1000000
SLIDING_WINDOW=131072
EXTRA_VOCAB_SIZE=293

moe_options=" \
            "

cpu_options=""

elif [ $MODEL_SIZE = 7B ]; then

HIDDEN_SIZE=3584
INTERMEDIATE_SIZE=18944
MAX_POSITION_EMBEDDINGS=131072
MAX_WINDOW_LAYERS=28
NUM_ATTENTION_HEADS=28
NUM_HIDDEN_LAYERS=28
NUM_KEY_VALUE_HEADS=4
RMS_NORM_EPS=1e-6
ROPE_THETA=1000000
SLIDING_WINDOW=131072
EXTRA_VOCAB_SIZE=293

moe_options=" \
            "

cpu_options=""

elif [ $MODEL_SIZE = 72B ]; then

HIDDEN_SIZE=8192
INTERMEDIATE_SIZE=29568
MAX_POSITION_EMBEDDINGS=131072
MAX_WINDOW_LAYERS=80
NUM_ATTENTION_HEADS=64
NUM_HIDDEN_LAYERS=80
NUM_KEY_VALUE_HEADS=8
RMS_NORM_EPS=1e-5
ROPE_THETA=1000000
SLIDING_WINDOW=131072
EXTRA_VOCAB_SIZE=421

moe_options=" \
            "
cpu_options=" \
            --use-cpu-initialization"

fi


if [ $MG2HF = true ]; then
    convert_options=" \
                --convert-checkpoint-from-megatron-to-transformers \
                --hf-ckpt-path ${HF_CKPT_PATH}"

elif [ $MG2HF = false ]; then
    convert_options=""
fi

if [ $USE_TE = true ]; then
    te_options=" \
                --transformer-impl transformer_engine \
                "

elif [ $USE_TE = false ]; then
    te_options=" \
                --transformer-impl local \
                "
fi


DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

torchrun ${DISTRIBUTED_ARGS} hf2mcore_qwen2_dense_gqa.py \
    --load ${SOURCE_CKPT_PATH} \
    --save ${TARGET_CKPT_PATH} \
    --target-tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --micro-batch-size 1 \
    --save-interval 1 \
    --bf16 \
    --swiglu \
    --num-layers ${NUM_HIDDEN_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --ffn-hidden-size ${INTERMEDIATE_SIZE} \
    --num-attention-heads ${NUM_ATTENTION_HEADS} \
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
    --seq-length 1 \
    --no-async-tensor-model-parallel-allreduce \
    --patch-tokenizer-type Qwen2Tokenizer \
    --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
    --untie-embeddings-and-output-weights \
    --no-bias-swiglu-fusion \
    --no-rope-fusion \
    --use-rotary-position-embeddings \
    --disable-bias-linear \
    --add-qkv-bias \
    --group-query-attention \
    --num-query-groups ${NUM_KEY_VALUE_HEADS} \
    --normalization RMSNorm \
    --norm-epsilon ${RMS_NORM_EPS} \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --rotary-base ${ROPE_THETA} \
    ${moe_options} \
    ${te_options} \
    ${convert_options} \
    ${cpu_options}


ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"