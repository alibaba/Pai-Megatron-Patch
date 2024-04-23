#!/bin/bash

# hf2megablocks: tp1_pp1_ep8_exp8_top2

# sh hf2megablocks_convertor_1.5.sh 0.5B /mnt/qwen-ckpts/Qwen1.5-0.5B ../../../ /mnt/qwen-ckpts/Qwen1.5-0.5B /mnt/qwen-ckpts/Qwen1.5-0.5B_megablocks_tp1_pp1_ep8_exp8 1 1 293 8 8 2 false

set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
START_TIME=$SECONDS
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8

MODEL_SIZE=$1
HG_CKPT_PATH=$2
MEGATRON_PATH=$3
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATH}:${MEGATRON_PATH}/Megatron-LM-MegaBlocks
SOURCE_CKPT_PATH=$4
TARGET_CKPT_PATH=$5
TP=$6
PP=$7
EXTRA_VOCAB_SIZE=$8
NUM_EXPERTS=$9
EXPERTS_TOPK=${10}
EP=${11}
mg2hf=${12}

if [ $MODEL_SIZE = 0.5B ]; then

NUM_LAYERS=24
HIDDEN_SIZE=1024
NUM_ATTN_HEADS=16
INTERMEDIATE_SIZE=2816

elif [ $MODEL_SIZE = 1.8B ]; then

NUM_LAYERS=24
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
INTERMEDIATE_SIZE=5504

elif [ $MODEL_SIZE = 7B ]; then

NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
INTERMEDIATE_SIZE=11008

elif [ $MODEL_SIZE = 14B ]; then

NUM_LAYERS=40
HIDDEN_SIZE=5120
NUM_ATTN_HEADS=40
INTERMEDIATE_SIZE=13696

fi


if [ $NUM_EXPERTS -gt 0 ]; then
    expert_options="
                --moe-top-k ${EXPERTS_TOPK} \
                --moe-num-experts ${NUM_EXPERTS} \
                --moe-expert-model-parallelism \
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

template_json="./hf_qwen1.5_moe/config_TEMPLATE.json"
config_json="./hf_qwen1.5_moe/config.json"
sed "s/CONFIG_HIDDEN_SIZE/${HIDDEN_SIZE}/" ${template_json} \
    | sed "s/CONFIG_INTERMEDIATE_SIZE/${INTERMEDIATE_SIZE}/" \
    | sed "s/CONFIG_ATTENTION_HEADS/${NUM_ATTN_HEADS}/" \
    | sed "s/CONFIG_HIDDEN_LAYERS/${NUM_LAYERS}/" \
    | sed "s/CONFIG_NUM_EXPERTS/${NUM_EXPERTS}/" \
    | sed "s/CONFIG_EXPERTS_topk/${EXPERTS_TOPK}/" \
    | sed "s/CONFIG_KV_HEADS/${NUM_ATTN_HEADS}/" \
	  > ${config_json}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
torchrun ${DISTRIBUTED_ARGS} hf2megablocks_qwen1.5.py \
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
    --norm-epsilon 1e-6 \
    --num-attention-heads 1 \
    --max-position-embeddings 1 \
    --seq-length 1 \
    --no-async-tensor-model-parallel-allreduce \
    --patch-tokenizer-type Qwen2Tokenizer \
    --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
    --untie-embeddings-and-output-weights \
    --use-llama2-rotary-position-embeddings \
    --disable-bias-linear \
    --normalization RMSNorm \
    --add-qkv-bias \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    ${expert_options} \
    ${convert_options}

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"