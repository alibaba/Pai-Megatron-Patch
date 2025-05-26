#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true # for PyTorch >= 2.6
START_TIME=$SECONDS
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

if [ -z ${MP_SAVE_SAFE_TENSORS} ];then
    MP_SAVE_SAFE_TENSORS=false
fi

if [ ${MP_SAVE_SAFE_TENSORS} = true ];then
    safe_options=" \
        --save-safetensors"
else
    safe_options=""
fi

if [ -z ${MP_VP} ]; then
    vp_options=""
else
    vp_options=" \
        --target-num-layers-per-virtual-pipeline-stage ${MP_VP}"
fi

MODEL_SIZE=$1
SOURCE_CKPT_PATH=$2
TARGET_CKPT_PATH=$3
TP=$4
PP=$5
MG2HF=$6
PR=$7
HF_CKPT_PATH=$8

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATCH_PATH=$( dirname $(dirname $( dirname ${CURRENT_DIR})))
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATCH_PATH}:${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-250217

USE_TE=true
if [ $MODEL_SIZE = 3B ]; then

NUM_LAYERS=36 # num_hidden_layers
HIDDEN_SIZE=2048 # hidden_size
NUM_ATTN_HEADS=16 # num_attention_heads
INTERMEDIATE_SIZE=11008 # intermediate_size
NUM_KEY_VALUE_HEADS=2 # num_key_value_heads
MAX_POSITION_EMBEDDINGS=128000 # max_position_embeddings
EXTRA_VOCAB_SIZE=293 # 151643 + 293 = 151936
RMS_NORM_EPS=1e-6 # rms_norm_eps


gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"


tie_option="" # tie_word_embeddings


elif [ $MODEL_SIZE = 7B ]; then

NUM_LAYERS=28
HIDDEN_SIZE=3584
NUM_ATTN_HEADS=28
INTERMEDIATE_SIZE=18944
NUM_KEY_VALUE_HEADS=4
MAX_POSITION_EMBEDDINGS=128000
EXTRA_VOCAB_SIZE=421  # 151643 + 421 = 152064
RMS_NORM_EPS=1e-6

gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"


tie_option=" \
        --untie-embeddings-and-output-weights \
        "

elif [ $MODEL_SIZE = 32B ]; then

NUM_LAYERS=64
HIDDEN_SIZE=5120
NUM_ATTN_HEADS=40
INTERMEDIATE_SIZE=27648
NUM_KEY_VALUE_HEADS=8
MAX_POSITION_EMBEDDINGS=128000
EXTRA_VOCAB_SIZE=421  # 151643 + 421 = 152064
RMS_NORM_EPS=1e-6

gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"


tie_option=" \
        --untie-embeddings-and-output-weights \
        "

elif [ $MODEL_SIZE = 72B ]; then

NUM_LAYERS=80
HIDDEN_SIZE=8192
NUM_ATTN_HEADS=64
INTERMEDIATE_SIZE=29568
NUM_KEY_VALUE_HEADS=8
MAX_POSITION_EMBEDDINGS=128000
EXTRA_VOCAB_SIZE=421
RMS_NORM_EPS=1e-6

gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"


tie_option=" \
        --untie-embeddings-and-output-weights \
        "


fi

if [ $MG2HF = true ]; then
    convert_options=" \
                --convert-checkpoint-from-megatron-to-transformers \
                --hf-ckpt-path ${HF_CKPT_PATH}"
    PRETRAIN_CHECKPOINT_PATH=${HF_CKPT_PATH}
elif [ $MG2HF = false ]; then
    convert_options=""
    PRETRAIN_CHECKPOINT_PATH=${SOURCE_CKPT_PATH}
fi

te_options=" \
            --transformer-impl transformer_engine \
            "

if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16"

elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"

fi

cpu_options=" \
            --use-cpu-initialization"
DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

if [ $PP -gt 1 ]; then
    tie_option=" \
        --untie-embeddings-and-output-weights \
        "
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

mkdir -p ${TARGET_CKPT_PATH}
# NOTE: model.safetensors.index.json will be copied by the following line and 
# should be removed in mg->hf conversion if save_safetensor is disabled.
find -L ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${TARGET_CKPT_PATH}
find -L ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "merges.txt" -print0 | xargs -0 cp -t ${TARGET_CKPT_PATH}

cmd="torchrun ${DISTRIBUTED_ARGS} hf2mcore_qwen2.5_vl.py \
    --load ${SOURCE_CKPT_PATH} \
    --save ${TARGET_CKPT_PATH} \
    --target-tensor-model-parallel-size ${TP} \
    --target-pipeline-model-parallel-size ${PP} \
    --micro-batch-size 1 \
    --save-interval 1 \
    --swiglu \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --ffn-hidden-size ${INTERMEDIATE_SIZE} \
    --num-attention-heads ${NUM_ATTN_HEADS} \
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
    --seq-length 1 \
    --no-async-tensor-model-parallel-allreduce \
    --patch-tokenizer-type Qwen2Tokenizer \
    --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
    --no-bias-swiglu-fusion \
    --no-rope-fusion \
    --use-rotary-position-embeddings \
    --disable-bias-linear \
    --add-qkv-bias \
    --normalization RMSNorm \
    --norm-epsilon ${RMS_NORM_EPS} \
    --use-mcore-models \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --rotary-base 1000000 \
    --spatial-merge-size 2 \
    ${safe_options} \
    ${te_options} \
    ${convert_options} \
    ${pr_options} \
    ${cpu_options} \
    ${tie_option} \
    ${gqa_options} \
    ${uneven_split_option} \
    ${vp_options}"

echo $cmd
eval $cmd


ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"