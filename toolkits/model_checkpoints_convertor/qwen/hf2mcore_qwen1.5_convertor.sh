#!/bin/bash
# hf2mcore: tp1_pp1
# sh hf2mcore_qwen1.5_convertor.sh 0.5B /mnt/qwen-ckpts/Qwen1.5-0.5B ../../../ /mnt/qwen-ckpts/Qwen1.5-0.5B /mnt/qwen-ckpts/Qwen1.5-0.5B-hf-to-mcore-tp1-pp1 1 1 293 0 0 0 false

# hf2mcore-moe: tp1_pp1_ep1_exp8
# sh hf2mcore_qwen1.5_convertor.sh 0.5B /mnt/qwen-ckpts/Qwen1.5-0.5B ../../../ /mnt/qwen-ckpts/Qwen1.5-0.5B /mnt/qwen-ckpts/Qwen1.5-0.5B-hf-to-mcore-tp1-pp1-ep1-exp8 1 1 293 8 2 1 false

set -e
export CUDA_VISIBLE_DEVICES=7
START_TIME=$SECONDS
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

MODEL_SIZE=$1
HG_CKPT_PATH=$2
MEGATRON_PATH=$3
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATH}:${MEGATRON_PATH}/Megatron-LM-240405
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

gqa_options=""
cpu_options=""

elif [ $MODEL_SIZE = 1.8B ]; then

NUM_LAYERS=24
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
INTERMEDIATE_SIZE=5504

gqa_options=""
cpu_options=""

elif [ $MODEL_SIZE = 7B ]; then

NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
INTERMEDIATE_SIZE=11008

gqa_options=""
cpu_options=""

elif [ $MODEL_SIZE = 14B ]; then

NUM_LAYERS=40
HIDDEN_SIZE=5120
NUM_ATTN_HEADS=40
INTERMEDIATE_SIZE=13696

gqa_options=""
cpu_options=""

elif [ $MODEL_SIZE = 32B ]; then

NUM_LAYERS=64
HIDDEN_SIZE=5120
NUM_ATTN_HEADS=40
INTERMEDIATE_SIZE=27392

cpu_options=""
gqa_options=" \
		    --group-query-attention \
		    --num-query-groups 8"

elif [ $MODEL_SIZE = 72B ]; then

NUM_LAYERS=80
HIDDEN_SIZE=8192
NUM_ATTN_HEADS=64
INTERMEDIATE_SIZE=24576

gqa_options=""
cpu_options=" \
		    --use-cpu-initialization"

fi


if [ $NUM_EXPERTS -gt 0 ]; then
    expert_options="
                --moe-router-topk ${EXPERTS_TOPK} \
                --num-experts ${NUM_EXPERTS} \
                --expert-model-parallel-size 1 \
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

DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

if [ $MODEL_SIZE != 32B ]; then

torchrun ${DISTRIBUTED_ARGS} hf2mcore_qwen1.5.py \
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
    --no-rope-fusion \
    --use-rotary-position-embeddings \
    --transformer-impl transformer_engine \
    --disable-bias-linear \
    --normalization RMSNorm \
    --add-qkv-bias \
    --use-mcore-models \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    ${expert_options} \
    ${convert_options} \
    ${gqa_options} \
    ${cpu_options}

else
python hf2mcore_qwen1.5_gqa.py \
  --load ${HG_CKPT_PATH} \
  --megatron-path ${MEGATRON_PATH} \
  --load_path ${SOURCE_CKPT_PATH} \
  --save_path ${TARGET_CKPT_PATH} \
  --target_params_dtype bf16 \
  --target_tensor_model_parallel_size ${TP} \
  --target_pipeline_model_parallel_size ${PP} \
${convert_options} \

fi

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"