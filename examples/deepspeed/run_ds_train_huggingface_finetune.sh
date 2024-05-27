#!/bin/bash

# bash run_ds_train_huggingface_finetune.sh \
#     --env dsw \
#     --model-size 13B \
#     --micro-batch-size 1 \
#     --gradient-accumulation-steps 2 \
#     --learning-rate 1e-5 \
#     --sequence-length 2048 \
#     --precision bf16 \
#     --zero-stage 2 \
#     --enable-gradient-checkpointing true \
#     --model-name llama2-13b \
#     --flash-attention true \
#     --epoch 2 \
#     --train-dataset /mnt/llama2-datasets/wudao_train.json \
#     --validation-dataset /mnt/llama2-datasets/wudao_valid.json \
#     --pretrain-model-path /mnt/llama2-ckpts/Llama-2-13b-hf \
#     --finetune-output-path /mnt/output_llama2_finetune

function usage() {
    echo '
Usage: bash run_ds_train_huggingface_finetune.sh \
    [--env ENV default dsw] \
    [--model-size MODEL_SIZE] \
    [--micro-batch-size MICRO_BATCH_SIZE default 1] \
    [--gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS default 1] \
    [--learning-rate LEARNING_RATE default 1e-5] \
    [--sequence-length SEQUENCE_LENGTH default 2048] \
    [--precision PRECISION default bf16] \
    [--zero-stage ZERO_STAGE default 2] \
    [--enable-gradient-checkpointing ENABLE_GRADIENT_CHECKPOINTING default true] \
    [--model-name MODEL_NAME {llama2-13b, qwen-7b, qwen-14b, qwen-72b}] \
    [--flash-attention FLASH_ATTENTION default false] \
    [--epoch EPOCH default 1] \
    [--train-dataset TRAIN_DATASET] \
    [--validation-dataset VALIDATION_DATASET] \
    [--pretrain-model-path PRETRAIN_MODEL_PATH] \
    [--finetune-output-path FINETUNE_OUTPUT_PATH]
'
}

ENV="dsw"
MODEL_SIZE=""
MICRO_BATCH_SIZE="1"
GRADIENT_ACCUMULATION_STEPS="1"
LEARNING_RATE="1e-5"
SEQUENCE_LENGTH="2048"
PRECISION="bf16"
ZERO_STAGE="2"
ENABLE_GRADIENT_CHECKPOINTING="true"
MODEL_NAME=""       # llama2-13b, qwen-7b, qwen-14b, qwen1.5-32b, qwen-72b
FLASH_ATTENTION="false"
EPOCH="1"
TRAIN_DATASET=""
VALIDATION_DATASET=""
PRETRAIN_MODEL_PATH=""
FINETUNE_OUTPUT_PATH=""
while [[ "$1" != "" ]]; do
    case $1 in
        --env )
            shift
            ENV=$1
            ;;
        --model-size )
            shift
            MODEL_SIZE=$1
            ;;
        --micro-batch-size )
            shift
            MICRO_BATCH_SIZE=$1
            ;;
        --gradient-accumulation-steps )
            shift
            GRADIENT_ACCUMULATION_STEPS=$1
            ;;
        --model-name )
            shift
            MODEL_NAME=$1
            ;;
        --learning-rate )
            shift
            LEARNING_RATE=$1
            ;;
        --sequence-length )
            shift
            SEQUENCE_LENGTH=$1
            ;;
        --precision )
            shift
            PRECISION=$1
            ;;
        --zero-stage )
            shift
            ZERO_STAGE=$1
            ;;
        --enable-gradient-checkpointing )
            shift
            ENABLE_GRADIENT_CHECKPOINTING=$1
            ;;
        --flash-attention )
            shift
            FLASH_ATTENTION=$1
            ;;
        --epoch )
            shift
            EPOCH=$1
            ;;
        --train-dataset )
            shift
            TRAIN_DATASET=$1
            ;;
        --validation-dataset )
            shift
            VALIDATION_DATASET=$1
            ;;
        --pretrain-model-path )
            shift
            PRETRAIN_MODEL_PATH=$1
            ;;
        --finetune-output-path )
            shift
            FINETUNE_OUTPUT_PATH=$1
            ;;
        -h | --help )
            usage
            exit 0
            ;;
        * )
            echo "Unknown argument ${1}"
            usage
            exit 1
            ;;
    esac
    shift
done

for i in "ENV=$ENV" \
    "MODEL_SIZE=$MODEL_SIZE" \
    "MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE" \
    "GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS" \
    "LEARNING_RATE=$LEARNING_RATE" \
    "SEQUENCE_LENGTH=$SEQUENCE_LENGTH" \
    "PRECISION=$PRECISION" \
    "ZERO_STAGE=$ZERO_STAGE" \
    "ENABLE_GRADIENT_CHECKPOINTING=$ENABLE_GRADIENT_CHECKPOINTING" \
    "FLASH_ATTENTION=$FLASH_ATTENTION" \
    "MODEL_NAME=$MODEL_NAME" \
    "EPOCH=$EPOCH" \
    "TRAIN_DATASET=$TRAIN_DATASET" \
    "VALIDATION_DATASET=$VALIDATION_DATASET" \
    "PRETRAIN_MODEL_PATH=$PRETRAIN_MODEL_PATH" \
    "PRETRAIN_MODEL_PATH=$PRETRAIN_MODEL_PATH" \
    "FINETUNE_OUTPUT_PATH=$FINETUNE_OUTPUT_PATH"
do
    config=(${i//=/ })
    config_name=${config[0]}
    config_value=${config[1]}
    if [ -z $config_value ]; then
        echo "$config_name is null"
        usage
        exit 1
    fi
done


set -e
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
GLOBAL_BATCH_SIZE=$(( ${MICRO_BATCH_SIZE} * ${GRADIENT_ACCUMULATION_STEPS} * ${GPUS_PER_NODE} * ${NNODES}))

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

elif [ $MODEL_SIZE = 32B ]; then

    NUM_LAYERS=64
    HIDDEN_SIZE=5120
    NUM_ATTN_HEADS=40
    INTERMEDIATE_SIZE=27392

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

if [ $PRECISION = fp16 ]; then
    pr_options="--fp16"
    FP16='true'
    BF16='false'
elif [ $PRECISION = bf16 ]; then
    pr_options="--bf16"
    FP16='false'
    BF16='true'
fi

if [ $ENABLE_GRADIENT_CHECKPOINTING = true ]; then
    gc_options="--enable-gradient-checkpointing"
elif [ $ENABLE_GRADIENT_CHECKPOINTING = false ]; then
    gc_options=""
fi

if [ $FLASH_ATTENTION = true ]; then
    flash_options="--flash"
elif [ $FLASH_ATTENTION = false ]; then
    flash_options=""
fi

NAME="${ENV}-ds-train-huggingface-finetune-${MODEL_SIZE}-lr-${LEARNING_RATE}-bs-${MICRO_BATCH_SIZE}-epoch-${EPOCH}-zero-${ZERO_STAGE}"
mkdir -p "${FINETUNE_OUTPUT_PATH}/tensorboard/"
mkdir -p "${FINETUNE_OUTPUT_PATH}/checkpoint/"
mkdir -p "${FINETUNE_OUTPUT_PATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
LOGGING_DIR="${FINETUNE_OUTPUT_PATH}/log/${NAME}_${current_time}"
mkdir -p ${LOGGING_DIR}

FINETUNE_CHECKPOINT_PATH="${FINETUNE_OUTPUT_PATH}/checkpoint/${NAME}"

hf_options="  \
        --load ${PRETRAIN_MODEL_PATH} \
        --save ${FINETUNE_CHECKPOINT_PATH} \
        --train-data ${TRAIN_DATASET} \
        --valid-data ${VALIDATION_DATASET} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --seq-length ${SEQUENCE_LENGTH} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --intermediate-size ${INTERMEDIATE_SIZE} \
        --micro-batch-size ${MICRO_BATCH_SIZE} \
        --epochs ${EPOCH} \
        --lr ${LEARNING_RATE} \
        --num-workers 1 \
        --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} \
        --logging-dir ${LOGGING_DIR} \
        --model ${MODEL_NAME} \
        ${pr_options} \
        ${gc_options} \
        ${flash_options}
        "

template_json="ds_config_TEMPLATE.json"
config_json="ds_config.json"
sed "s/CONFIG_MBSIZE/${MICRO_BATCH_SIZE}/" ${template_json} \
    | sed "s/CONFIG_ZERO_STATE/${ZERO_STAGE}/" \
    | sed "s/CONFIG_GBSIZE/${GLOBAL_BATCH_SIZE}/" \
    | sed "s/CONFIG_GAS/${GRADIENT_ACCUMULATION_STEPS}/" \
    | sed "s/CONFIG_FP16_ENABLED/${FP16}/" \
    | sed "s/CONFIG_BF16_ENABLED/${BF16}/" \
    | sed "s/CONFIG_LR/${LEARNING_RATE}/" \
	  > ${config_json}

run_cmd="torchrun $DISTRIBUTED_ARGS ds_train_huggingface_finetune.py ${hf_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
