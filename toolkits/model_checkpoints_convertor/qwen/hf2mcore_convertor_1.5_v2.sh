#!/bin/bash
# hf2mg: tp1_pp1
# sh hf2mcore_convertor_1.5_v2.sh 0.5B /mnt/qwen-ckpts/Qwen1.5-0.5B ../../../ /mnt/qwen-ckpts/Qwen1.5-0.5B /mnt/qwen-ckpts/qwen1.5_0.5b_mcore_tp1_pp1_v2 1 1 293 0 0 0 false

# hf2mg: tp1_pp1_ep1_exp8
# sh hf2mcore_convertor_1.5_v2.sh 0.5B /mnt/qwen-ckpts/Qwen1.5-1.8B ../../../ /mnt/qwen-ckpts/Qwen1.5-1.8B /mnt/qwen-ckpts/Qwen1.5-1.8B_mcore_tp1_pp1_ep1_exp8_v2 1 1 293 8 1 2 false

# hf2mg: tp1_pp1_ep2_exp8
# sh hf2mcore_convertor_1.5_v2.sh 0.5B /mnt/qwen-ckpts/Qwen1.5-1.8B ../../../ /mnt/qwen-ckpts/Qwen1.5-1.8B /mnt/qwen-ckpts/Qwen1.5-1.8B_mcore_tp1_pp1_ep2_exp8_v2 1 1 293 8 2 false

# hf2mg: tp2_pp1_ep4_exp8
#sh hf2mcore_convertor_1.5_v2.sh 0.5B /mnt/qwen-ckpts/Qwen1.5-1.8B ../../../ /mnt/qwen-ckpts/Qwen1.5-1.8B /mnt/qwen-ckpts/Qwen1.5-1.8B_mcore_tp2_pp1_ep4_exp8_v2 2 1 293 8 2 4 false

# hf2mg: tp2_pp1_ep4_exp32
#sh hf2mcore_convertor_1.5_v2.sh 0.5B /mnt/qwen-ckpts/Qwen1.5-1.8B ../../../ /mnt/qwen-ckpts/Qwen1.5-1.8B /mnt/qwen-ckpts/Qwen1.5-1.8B_mcore_tp2_pp1_ep4_exp32_v2 2 1 293 32 4 4 false

# mg2hf: tp1_pp1_ep1_exp8
#sh hf2mcore_convertor_1.5_v2.sh 0.5B /mnt/qwen-ckpts/Qwen1.5-0.5B ../../../ /mnt/qwen-ckpts/Qwen1.5-0.5B_mcore_tp1_pp1_ep1_exp8_v2 /mnt/qwen-ckpts/mg2hf 1 1 293 8 1 true

# mg2hf: tp2_pp1_ep4_exp32
#sh hf2mcore_convertor_1.5_v2.sh 1.8B /mnt/qwen-ckpts/Qwen1.5-1.8B ../../../ /mnt/qwen-ckpts/test_qwen1_8B_32experts/ /mnt/qwen-ckpts/mg_test_qwen1_8B_32experts_auxloss 2 1 293 32 4 4 true

set -e
export CUDA_VISIBLE_DEVICES=3
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

template_json="./hf_qwen_moe/config_TEMPLATE.json"
config_json="./hf_qwen_moe/config.json"
sed "s/CONFIG_HIDDEN_SIZE/${HIDDEN_SIZE}/" ${template_json} \
    | sed "s/CONFIG_INTERMEDIATE_SIZE/${INTERMEDIATE_SIZE}/" \
    | sed "s/CONFIG_ATTENTION_HEADS/${NUM_ATTN_HEADS}/" \
    | sed "s/CONFIG_HIDDEN_LAYERS/${NUM_LAYERS}/" \
    | sed "s/CONFIG_NUM_EXPERTS/${NUM_EXPERTS}/" \
    | sed "s/CONFIG_EXPERTS_topk/${EXPERTS_TOPK}/" \
    | sed "s/CONFIG_KV_HEADS/${NUM_ATTN_HEADS}/" \
	  > ${config_json}

DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

torchrun ${DISTRIBUTED_ARGS} hf2mcore_1.5_v2.py \
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
    ${convert_options}

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"