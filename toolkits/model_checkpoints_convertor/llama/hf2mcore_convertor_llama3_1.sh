#!/bin/bash

set -e
if [ -z $CUDA_VISIBLE_DEVICES ];then
    export CUDA_VISIBLE_DEVICES=0
fi
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true # for PyTorch >= 2.6
if [ -z $NAIVE_CHECK ];then
    NAIVE_CHECK=false
fi

MEGATRON_PATCH_PATH=$( dirname $(dirname $( dirname ${CURRENT_DIR})))
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATCH_PATH}:${MEGATRON_PATCH_PATH}/backends/megatron/PAI-Megatron-LM-240718


START_TIME=$SECONDS
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

MODEL_SIZE=$1
SOURCE_CKPT_PATH=$2
TARGET_CKPT_PATH=$3
TP=$4
PP=$5
mg2hf=${6}
CHECK=${7}
CHECK_ONLY=${8}
PR=${9}

if [ $mg2hf = false ]; then
    HF_CKPT_PATH=$SOURCE_CKPT_PATH
else
    HF_CKPT_PATH=${10}
fi

if [ $MODEL_SIZE = 70B ]; then

NUM_LAYERS=80
HIDDEN_SIZE=8192
NUM_ATTN_HEADS=64
INTERMEDIATE_SIZE=28672
NUM_KV_HEADS=8
VOCAB_SIZE=128256
ROPE_THETA=500000
gqa_options=" \
		    --group-query-attention \
		    --num-query-groups 8"
EXTRA_VOCAB_SIZE=256

cpu_options=" \
            --use-cpu-initialization"

elif [ $MODEL_SIZE = 8B ]; then

NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
INTERMEDIATE_SIZE=14336
NUM_KV_HEADS=8
VOCAB_SIZE=128256
ROPE_THETA=500000

gqa_options=" \
		    --group-query-attention \
		    --num-query-groups 8"
EXTRA_VOCAB_SIZE=256

cpu_options=""

fi

if [ $mg2hf = true ]; then
    convert_options="
                --convert-checkpoint-from-megatron-to-transformers
    "
elif [ $mg2hf = false ]; then
    convert_options=""
fi


te_options=" \
            --transformer-impl transformer_engine \
            "



if [ $CHECK = true ]; then
    convert_options=" ${convert_options}\
                --check-alignment \
                "
fi
if [ $CHECK_ONLY = true ]; then
    convert_options=" ${convert_options}\
                --check-only \
                "
fi
if [ $NAIVE_CHECK = true ]; then
    convert_options=" ${convert_options}\
                --naive-check \
                "
fi

if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16"

elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
fi

DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
run_cmd="torchrun ${DISTRIBUTED_ARGS} hf2mcore_llama3_1.py \
    --load ${SOURCE_CKPT_PATH} \
    --save ${TARGET_CKPT_PATH} \
    --hf-ckpt-path ${HF_CKPT_PATH} \
    --target-tensor-model-parallel-size ${TP} \
    --target-pipeline-model-parallel-size ${PP} \
    --micro-batch-size 1 \
    --save-interval 1 \
    --swiglu \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --ffn-hidden-size ${INTERMEDIATE_SIZE} \
    --num-attention-heads ${NUM_ATTN_HEADS} \
    --norm-epsilon 1e-5 \
    --max-position-embeddings 1 \
    --no-bias-swiglu-fusion \
    --seq-length 1 \
    --no-async-tensor-model-parallel-allreduce \
    --patch-tokenizer-type LLamaTokenizer \
    --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
    --untie-embeddings-and-output-weights \
    --no-rope-fusion \
    --use-rotary-position-embeddings \
    --disable-bias-linear \
    --normalization RMSNorm \
    --use-mcore-models \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --vocab-size ${VOCAB_SIZE} \
    --rotary-base ${ROPE_THETA} \
    ${convert_options} \
    ${gqa_options} \
    ${pr_options} \
    ${cpu_options}"

echo ${run_cmd}
eval ${run_cmd}
ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"