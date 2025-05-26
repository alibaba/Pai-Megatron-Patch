#!/usr/bin/env bash

set -ex

export CUDA_VISIBLE_DEVICES=3
START_TIME=$SECONDS
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

MODEL_SIZE=$1
SOURCE_CKPT_PATH=$2
TARGET_CKPT_PATH=$3
TP=$4
PP=$5
ETP=$6
EP=$7
PR=$8
MG2HF=$9
HF_CKPT_PATH=${10}

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATCH_PATH=$( dirname $(dirname $( dirname ${CURRENT_DIR})))
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATCH_PATH}:${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-250328

if [ $MODEL_SIZE = 0.6B ]; then
    NUM_LAYERS=28
    HIDDEN_SIZE=1024
    NUM_ATTN_HEADS=16
    INTERMEDIATE_SIZE=3072
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    RMS_NORM_EPS=1e-6
    gqa_options=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
    
    tie_option=""
    moe_options=""
elif [ $MODEL_SIZE = 1.7B ]; then
    NUM_LAYERS=28
    HIDDEN_SIZE=2048
    NUM_ATTN_HEADS=16
    INTERMEDIATE_SIZE=6144
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    RMS_NORM_EPS=1e-6
    gqa_options=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
    
    tie_option=""
    moe_options=""
elif [ $MODEL_SIZE = 4B ]; then
    NUM_LAYERS=36
    HIDDEN_SIZE=2560
    NUM_ATTN_HEADS=32
    INTERMEDIATE_SIZE=9728
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    RMS_NORM_EPS=1e-6
    gqa_options=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
    
    tie_option=""
    moe_options=""
elif [ $MODEL_SIZE = 8B ]; then
    NUM_LAYERS=36
    HIDDEN_SIZE=4096
    NUM_ATTN_HEADS=32
    INTERMEDIATE_SIZE=12288
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    RMS_NORM_EPS=1e-6
    gqa_options=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
    
    tie_option=" \
            --untie-embeddings-and-output-weights \
            "
    moe_options=""
elif [ $MODEL_SIZE = 14B ]; then 
    NUM_LAYERS=40
    HIDDEN_SIZE=5120
    NUM_ATTN_HEADS=40
    INTERMEDIATE_SIZE=17408
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    RMS_NORM_EPS=1e-6
    gqa_options=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
    
    tie_option=" \
            --untie-embeddings-and-output-weights \
            "

    moe_options=""
elif [ $MODEL_SIZE = 32B ]; then
    NUM_LAYERS=64
    HIDDEN_SIZE=5120
    NUM_ATTN_HEADS=64
    INTERMEDIATE_SIZE=25600
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    RMS_NORM_EPS=1e-5
    gqa_options=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
    
    tie_option=" \
            --untie-embeddings-and-output-weights \
            "

    moe_options=""

    cpu_options=" \
            --use-cpu-initialization"

elif [ $MODEL_SIZE = A3B ]; then
    HIDDEN_SIZE=2048
    NUM_ATTENTION_HEADS=32
    NUM_LAYERS=48
    INTERMEDIATE_SIZE=6144
    MOE_INTERMEDIATE_SIZE=768
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    NUM_KEY_VALUE_HEADS=4
    ROPE_THETA=1000000
    NUM_EXPERTS=128
    ROUTER_TOPK=8
    RMS_NORM_EPS=1e-6

    moe_options=" \
        --moe-grouped-gemm \
        --moe-token-dispatcher-type alltoall \
        --moe-router-topk ${ROUTER_TOPK} \
        --num-experts ${NUM_EXPERTS} \
        --target-expert-tensor-parallel-size ${ETP} \
        --target-expert-model-parallel-size ${EP} \
        --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
        --moe-router-load-balancing-type aux_loss \
        --moe-aux-loss-coeff 0.001 \
        --moe-layer-freq '([1]*48)' \
        --moe-router-pre-softmax
        "

    tie_option=" \
            --untie-embeddings-and-output-weights \
            "

    gqa_options=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"

    cpu_options=" \
            --use-cpu-initialization"

elif [ $MODEL_SIZE = A22B ]; then
    HIDDEN_SIZE=4096
    NUM_ATTENTION_HEADS=64
    NUM_LAYERS=94
    INTERMEDIATE_SIZE=12288
    MOE_INTERMEDIATE_SIZE=1536
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    NUM_KEY_VALUE_HEADS=4
    ROPE_THETA=1000000
    NUM_EXPERTS=128
    ROUTER_TOPK=8
    RMS_NORM_EPS=1e-6


    moe_options=" \
        --moe-grouped-gemm \
        --moe-token-dispatcher-type alltoall \
        --moe-router-topk ${ROUTER_TOPK} \
        --num-experts ${NUM_EXPERTS} \
        --target-expert-tensor-parallel-size ${ETP} \
        --target-expert-model-parallel-size ${EP} \
        --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
        --moe-router-load-balancing-type aux_loss \
        --moe-aux-loss-coeff 0.001 \
        --moe-layer-freq '([1]*94)' \
        --moe-router-pre-softmax
        "

    tie_option=" \
            --untie-embeddings-and-output-weights \
            "

    gqa_options=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
    
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

if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16"

elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"

fi

if [ -z ${MP_PP0_LAYERS} ];then
    uneven_split_option=""
elif [ ${PP} -gt 1 ]; then
    _check=$(( ( $NUM_LAYERS - ${MP_PP0_LAYERS} ) % ( ${PP} - 1 ) ))
    if [ $_check != 0 ]; then
        echo "With uneven pipelineing the left over layers must be divisible by left over stages."
        exit -1
    fi

    uneven_split_option=" \
        --target-decoder-first-pipeline-num-layers ${MP_PP0_LAYERS}
    "
else
    echo "uneven pipeline split must be used when PP > 1"
    exit -1
fi


DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

torchrun ${DISTRIBUTED_ARGS} hf2mcore_qwen2_moe.py \
    --load ${SOURCE_CKPT_PATH} \
    --save ${TARGET_CKPT_PATH} \
    --target-tensor-model-parallel-size ${TP} \
    --target-pipeline-model-parallel-size ${PP} \
    --micro-batch-size 1 \
    --save-interval 1 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --swiglu \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --ffn-hidden-size ${INTERMEDIATE_SIZE} \
    --num-attention-heads ${NUM_ATTENTION_HEADS} \
    --max-position-embeddings 10 \
    --max-padding-length 10 \
    --seq-length 10 \
    --no-async-tensor-model-parallel-allreduce \
    --patch-tokenizer-type Qwen3Tokenizer \
    --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
    --no-bias-swiglu-fusion \
    --no-rope-fusion \
    --position-embedding-type rope \
    --use-rotary-position-embeddings \
    --disable-bias-linear \
    --normalization RMSNorm \
    --norm-epsilon ${RMS_NORM_EPS} \
    --rotary-base ${ROPE_THETA} \
    --rotary-seq-len-interpolation-factor 1 \
    --transformer-impl transformer_engine \
    --attention-backend fused \
    --dist-ckpt-strictness ignore_all \
    --qk-layernorm \
    --kv-channels 128 \
    ${moe_options} \
    ${convert_options} \
    ${pr_options} \
    ${uneven_split_option} \
    ${cpu_options} \
    ${gqa_options} \
    ${tie_option}

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"