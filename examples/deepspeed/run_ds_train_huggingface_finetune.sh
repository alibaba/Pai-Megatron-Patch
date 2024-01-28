#!/bin/bash
# bash run_ds_train_huggingface_finetune.sh dsw 7B 1 2 1e-5 2048 bf16 2 true qwen-7b false 2 /mnt/qwen-datasets/wudao_train.json /mnt/qwen-datasets/wudao_valid.json /mnt/qwen-ckpts/qwen-7b-hf /mnt/output_qwen_7b_finetune
# bash run_ds_train_huggingface_finetune.sh dsw 13B 1 2 1e-5 2048 bf16 2 true llama2-13b true 2 /mnt/llama2-datasets/wudao_train.json /mnt/llama2-datasets/wudao_valid.json /mnt/llama2-ckpts/Llama-2-13b-hf /mnt/output_llama2_finetune

set -e
ENV=$1
export CUDA_DEVICE_MAX_CONNECTIONS=1
if [ $ENV = dsw ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    MASTER_ADDR=localhost
    MASTER_PORT=$(shuf -n 1 -i 10000-65535)
    NNODES=1
    NODE_RANK=0
    GPUS_PER_NODE=8
elif [ $ENV = dlc ]; then
    NNODES=${WORLD_SIZE}
    NODE_RANK=${RANK}
    GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}
fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

MODEL_SIZE=$2
MICRO_BATCH_SIZE=$3
GA_STEPS=$4
LR=$5
SEQ_LEN=$6
PR=$7
ZERO=$8
GC=$9
MODEL=${10}         # llama2-13b, qwen-7b, qwen-14b, qwen-72b
FLASH=${11}
EPOCH=${12}
TRAIN_DATASET_PATH=${13}
VALID_DATASET_PATH=${14}
PRETRAIN_CHECKPOINT_PATH=${15}
OUTPUT_BASEPATH=${16}

GLOBAL_BATCH_SIZE=$(( ${MICRO_BATCH_SIZE} * ${GA_STEPS} * ${GPUS_PER_NODE} * ${NNODES}))

if [ $MODEL_SIZE = 7B ]; then

    NUM_LAYERS=32
    HIDDEN_SIZE=4096
    NUM_ATTN_HEADS=32
    INTERMEDIATE_SIZE=11008

elif [ $MODEL_SIZE = 13B ]; then

    NUM_LAYERS=40
    HIDDEN_SIZE=5120
    NUM_ATTN_HEADS=40
    INTERMEDIATE_SIZE=13824

elif [ $MODEL_SIZE = 14B ]; then

    NUM_LAYERS=40
    HIDDEN_SIZE=5120
    NUM_ATTN_HEADS=40
    INTERMEDIATE_SIZE=13696

elif [ $MODEL_SIZE = 65B ]; then

    NUM_LAYERS=80
    HIDDEN_SIZE=8192
    NUM_ATTN_HEADS=64
    INTERMEDIATE_SIZE=22016

elif [ $MODEL_SIZE = 70B ]; then

    NUM_LAYERS=80
    HIDDEN_SIZE=8192
    NUM_ATTN_HEADS=64
    INTERMEDIATE_SIZE=28672

fi

if [ $PR = fp16 ]; then
    pr_options="--fp16"
    FP16='true'
    BF16='false'
elif [ $PR = bf16 ]; then
    pr_options="--bf16"
    FP16='false'
    BF16='true'
fi

if [ $GC = true ]; then
    gc_options="--enable-gradient-checkpointing"
elif [ $GC = false ]; then
    gc_options=""
fi

if [ $FLASH = true ]; then
    flash_options="--flash"
elif [ $FLASH = false ]; then
    flash_options=""
fi

NAME="${ENV}-ds-train-huggingface-finetune-${MODEL_SIZE}-lr-${LR}-bs-${MICRO_BATCH_SIZE}-epoch-${EPOCH}-zero-${ZERO}"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
LOGGING_DIR="${OUTPUT_BASEPATH}/log/${NAME}_${current_time}"
mkdir -p ${LOGGING_DIR}

FINETUNE_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

hf_options="  \
        --load ${PRETRAIN_CHECKPOINT_PATH} \
        --save ${FINETUNE_CHECKPOINT_PATH} \
        --train-data ${TRAIN_DATASET_PATH} \
        --valid-data ${VALID_DATASET_PATH} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --seq-length ${SEQ_LEN} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --intermediate-size ${INTERMEDIATE_SIZE} \
        --micro-batch-size ${MICRO_BATCH_SIZE} \
        --epochs ${EPOCH} \
        --lr ${LR} \
        --num-workers 1 \
        --gradient-accumulation-steps ${GA_STEPS} \
        --logging-dir ${LOGGING_DIR} \
        --model ${MODEL} \
        ${pr_options} \
        ${gc_options} \
        ${flash_options}
        "

template_json="ds_config_TEMPLATE.json"
config_json="ds_config.json"
sed "s/CONFIG_MBSIZE/${MICRO_BATCH_SIZE}/" ${template_json} \
    | sed "s/CONFIG_ZERO_STATE/${ZERO}/" \
    | sed "s/CONFIG_GBSIZE/${GLOBAL_BATCH_SIZE}/" \
    | sed "s/CONFIG_GAS/${GA_STEPS}/" \
    | sed "s/CONFIG_FP16_ENABLED/${FP16}/" \
    | sed "s/CONFIG_BF16_ENABLED/${BF16}/" \
    | sed "s/CONFIG_LR/${LR}/" \
	  > ${config_json}

run_cmd="torchrun $DISTRIBUTED_ARGS ds_train_huggingface_finetune.py ${hf_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
