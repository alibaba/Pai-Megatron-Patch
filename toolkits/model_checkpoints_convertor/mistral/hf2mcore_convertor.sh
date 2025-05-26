#!/bin/bash

set -e
START_TIME=$SECONDS
export CUDA_VISIBLE_DEVICES=0
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true # for PyTorch >= 2.6
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

MODEL_SIZE=$1
HG_CKPT_PATH=$2
MEGATRON_PATH=$3
MEGATRON_PATCH_PATH=$( dirname $(dirname $( dirname ${CURRENT_DIR})))
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATCH_PATH}:${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-241113
SOURCE_CKPT_PATH=$4
TARGET_CKPT_PATH=$5
TP=$6
PP=$7
EXTRA_VOCAB_SIZE=$8
NUM_EXPERTS=$9
EXPERTS_TOPK=${10}
EP=${11}
mg2hf=${12}

if [ $MODEL_SIZE = 7B ]; then

NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
INTERMEDIATE_SIZE=14336
NUM_KEY_VALUE_HEADS=8

gqa_options=" \
		    --group-query-attention \
		    --num-query-groups 8"

elif [ $MODEL_SIZE = 8x7B ]; then

NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
INTERMEDIATE_SIZE=14336
NUM_KEY_VALUE_HEADS=8
WS=${13}
gqa_options=" \
		    --group-query-attention \
		    --num-query-groups 8"

fi


if [ $NUM_EXPERTS -gt 0 ]; then
    expert_options="
                --moe-router-topk ${EXPERTS_TOPK} \
                --num-experts ${NUM_EXPERTS} \
                --target_expert_model_parallel_size ${EP}
    "
fi

if [ $mg2hf = true ]; then
    convert_options="
                --convert_checkpoint_from_megatron_to_transformers
    "
elif [ $mg2hf = false ]; then
    convert_options=""
fi

template_json="./hf_mistral_moe/config_TEMPLATE.json"
config_json="./hf_mistral_moe/config.json"
sed "s/CONFIG_HIDDEN_SIZE/${HIDDEN_SIZE}/" ${template_json} \
    | sed "s/CONFIG_INTERMEDIATE_SIZE/${INTERMEDIATE_SIZE}/" \
    | sed "s/CONFIG_ATTENTION_HEADS/${NUM_ATTN_HEADS}/" \
    | sed "s/CONFIG_HIDDEN_LAYERS/${NUM_LAYERS}/" \
    | sed "s/CONFIG_NUM_EXPERTS/${NUM_EXPERTS}/" \
    | sed "s/CONFIG_EXPERTS_topk/${EXPERTS_TOPK}/" \
    | sed "s/CONFIG_KV_HEADS/${NUM_KEY_VALUE_HEADS}/" \
	  > ${config_json}

if [ $MODEL_SIZE = 7B ]; then

torchrun ${DISTRIBUTED_ARGS} hf2mcore.py \
    --load_path ${SOURCE_CKPT_PATH} \
    --save_path ${TARGET_CKPT_PATH} \
    --load ${HG_CKPT_PATH} \
    --huggingface_model_path ${HG_CKPT_PATH} \
    --megatron-path ${MEGATRON_PATH} \
    --target_tensor_model_parallel_size ${TP} \
    --target_pipeline_model_parallel_size ${PP} \
    --micro-batch-size 1 \
    --fp16 \
    --swiglu \
    --num-layers 1 \
    --hidden-size 1 \
    --ffn-hidden-size 1 \
    --norm-epsilon 1e-5 \
    --num-attention-heads 1 \
    --max-position-embeddings 1 \
    --seq-length 1 \
    --no-async-tensor-model-parallel-allreduce \
    --patch-tokenizer-type LLamaTokenizer \
    --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
    --untie-embeddings-and-output-weights \
    --no-rope-fusion \
    --use-rotary-position-embeddings \
    --transformer-impl transformer_engine \
    --disable-bias-linear \
    --normalization RMSNorm \
    --use-mcore-models \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    ${expert_options} \
    ${convert_options} \
    ${gqa_options} \

elif [ $MODEL_SIZE = 8x7B ]; then

python hf2mcore_mixtral.py \
--megatron-path ${MEGATRON_PATH} \
--load_path ${SOURCE_CKPT_PATH} \
--save_path ${TARGET_CKPT_PATH} \
--target_params_dtype bf16 \
--target_tensor_model_parallel_size ${TP} \
--target_pipeline_model_parallel_size ${PP} \
--target_expert_model_parallel_size ${EP} \
--world_size ${WS} \
${convert_options} \

fi

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"