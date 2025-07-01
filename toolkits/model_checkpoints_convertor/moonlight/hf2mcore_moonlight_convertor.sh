#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true # for PyTorch >= 2.6
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
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATCH_PATH}:${MEGATRON_PATCH_PATH}/backends/megatron//Megatron-LM-250328

if [ $MODEL_SIZE = A3B ]; then
# moonshotai/Moonlight-16B-A3B-Instruct
HIDDEN_SIZE=2048
NUM_ATTENTION_HEADS=16
NUM_LAYERS=27
INTERMEDIATE_SIZE=11264
MOE_INTERMEDIATE_SIZE=1408
MAX_POSITION_EMBEDDINGS=8192
EXTRA_VOCAB_SIZE=0
Q_LORA_RANK=0
KV_LORA_RANK=512
QK_NOPE_HEAD_DIM=128
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=128
ROPE_THETA=50000
SCALE_FACTOR=1
NUM_EXPERTS=64
ROUTER_TOPK=6
NUM_SHARED_EXPERTS=2
RMS_NORM_EPS=1e-5

moe_options=" \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall \
    --moe-router-topk ${ROUTER_TOPK} \
    --moe-router-group-topk 1 \
    --moe-router-num-groups 1 \
    --num-experts ${NUM_EXPERTS} \
    --target-expert-model-parallel-size ${EP} \
    --target-expert-tensor-parallel-size ${ETP} \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-router-topk-scaling-factor 2.446 \
    --moe-shared-expert-overlap \
    --moe-router-enable-expert-bias \
    --mscale 1.0 \
    --mscale-all-dim 1.0 \
    --moe-router-score-function sigmoid \
    --moe-router-bias-update-rate 0.001 \
    --moe-aux-loss-coeff 0.001 \
    --moe-layer-freq '([0]*1+[1]*26)' \
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-nope-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-rope-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    "

cpu_options=""

mtp_options=""
fi


if [ $MG2HF = true ]; then
    convert_options=" \
                --convert-checkpoint-from-megatron-to-transformers \
                --hf-ckpt-path ${HF_CKPT_PATH}"

    mkdir -p ${TARGET_CKPT_PATH}
    find -L ${HF_CKPT_PATH} -maxdepth 1 -type f -name "configuration.json" -print0 | xargs -0 cp -t ${TARGET_CKPT_PATH}
    find -L ${HF_CKPT_PATH} -maxdepth 1 -type f -name "tiktoken.model" -print0 | xargs -0 cp -t ${TARGET_CKPT_PATH}
    find -L ${HF_CKPT_PATH} -maxdepth 1 -type f -name "tiktoken.model" -print0 | xargs -0 cp -t ${SOURCE_CKPT_PATH}
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

cmd="torchrun ${DISTRIBUTED_ARGS} ../deepseek/hf2mcore_deepseek_v3_moe.py \
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
    --patch-tokenizer-type DeepSeekV2Tokenizer \
    --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
    --untie-embeddings-and-output-weights \
    --no-bias-swiglu-fusion \
    --no-rope-fusion \
    --rope-type rope \
    --position-embedding-type rope \
    --use-rotary-position-embeddings \
    --disable-bias-linear \
    --normalization RMSNorm \
    --norm-epsilon ${RMS_NORM_EPS} \
    --rotary-base ${ROPE_THETA} \
    --rotary-scaling-factor ${SCALE_FACTOR} \
    --rotary-seq-len-interpolation-factor 1 \
    --kv-channels ${V_HEAD_DIM} \
    --qk-layernorm \
    --multi-latent-attention \
    --transformer-impl transformer_engine \
    --attention-backend fused \
    --use-rope-scaling \
    --no-initialization \
    ${mtp_options} \
    ${moe_options} \
    ${convert_options} \
    ${pr_options} \
    ${uneven_split_option} \
    ${cpu_options}"

echo $cmd
eval $cmd
ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"