#!/bin/bash

set -e
export CUDA_VISIBLE_DEVICES=7
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true # for PyTorch >= 2.6
START_TIME=$SECONDS
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

MODEL_SIZE=$1
HG_CKPT_PATH=$2
MEGATRON_PATH=$3
MEGATRON_PATCH_PATH=$( dirname $(dirname $( dirname ${CURRENT_DIR})))
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATCH_PATH}:${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-240126
SOURCE_CKPT_PATH=$4
TARGET_CKPT_PATH=$5
TP=$6
PP=$7
EXTRA_VOCAB_SIZE=$8
NUM_EXPERTS=$9
EXPERTS_TOPK=${10}
EP=${11}
NUM_EXPERT_SPLITS=${12}
mg2hf=${13}

if [ $MODEL_SIZE = 7B ]; then

NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
INTERMEDIATE_SIZE=11008
NUM_KV_HEADS=32
VOCAB_SIZE=32000
ROPE_THETA=10000

gqa_options=""

elif [ $MODEL_SIZE = 13B ]; then

NUM_LAYERS=40
HIDDEN_SIZE=5120
NUM_ATTN_HEADS=40
INTERMEDIATE_SIZE=13824
NUM_KV_HEADS=40
VOCAB_SIZE=32000
ROPE_THETA=10000
gqa_options=""

elif [ $MODEL_SIZE = 70B ]; then

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

fi

if [ $NUM_EXPERT_SPLITS -gt 0 ]; then

INTERMEDIATE_SIZE=$(( ${INTERMEDIATE_SIZE} / ${NUM_EXPERT_SPLITS}))

fi

if [ $NUM_EXPERTS -gt 0 ]; then
    expert_options="
                --moe-router-topk ${EXPERTS_TOPK} \
                --num-experts ${NUM_EXPERTS} \
                --expert-model-parallel-size 1 \
                --target_expert_model_parallel_size ${EP} \
                --num_expert_split_size ${NUM_EXPERT_SPLITS} \
    "
fi

if [ $mg2hf = true ]; then
    convert_options="
                --convert_checkpoint_from_megatron_to_transformers
    "
elif [ $mg2hf = false ]; then
    convert_options=""
fi

template_json="./hf_llama_moe/config_TEMPLATE.json"
config_json="./hf_llama_moe/config.json"
sed "s/CONFIG_HIDDEN_SIZE/${HIDDEN_SIZE}/" ${template_json} \
    | sed "s/CONFIG_INTERMEDIATE_SIZE/${INTERMEDIATE_SIZE}/" \
    | sed "s/CONFIG_ATTENTION_HEADS/${NUM_ATTN_HEADS}/" \
    | sed "s/CONFIG_HIDDEN_LAYERS/${NUM_LAYERS}/" \
    | sed "s/CONFIG_NUM_EXPERTS/${NUM_EXPERTS}/" \
    | sed "s/CONFIG_EXPERTS_topk/${EXPERTS_TOPK}/" \
    | sed "s/CONFIG_KV_HEADS/${NUM_KV_HEADS}/" \
    | sed "s/CONFIG_VOCAB_SIZE/${VOCAB_SIZE}/" \
    | sed "s/CONFIG_ROPE_THETA/${ROPE_THETA}/" \
	  > ${config_json}

DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

if [ $MODEL_SIZE != 70B ]; then

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
    ${expert_options} \
    ${convert_options} \
    ${gqa_options} \

else
python hf2mcore_70b.py \
  --load ${HG_CKPT_PATH} \
  --megatron-path ${MEGATRON_PATH} \
  --load_path ${SOURCE_CKPT_PATH} \
  --save_path ${TARGET_CKPT_PATH} \
  --target_params_dtype bf16 \
  --target_tensor_model_parallel_size ${TP} \
  --target_pipeline_model_parallel_size ${PP} \
${convert_options} \
${cpu_options}

fi

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"