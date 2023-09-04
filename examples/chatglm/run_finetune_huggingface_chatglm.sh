#!/bin/bash
#sh run_finetune_huggingface_chatglm.sh dsw /workspace/PAI-Megatron-Patch/Megatron-LM/ /workspace/PAI-Megatron-Patch/ 6B 4 64 64 1e-4 1e-5 fp16 true /mnt/glm-datasets/AdvertiseGen/train.json /mnt/glm-datasets/AdvertiseGen/dev.json /mnt/glm-ckpts/chatglm-6b 2 /mnt/output_megatron_chatglm
set -e
ENV=$1
MEGATRON_PATH=$2
MEGATRON_PATCH_PATH=$3
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:$PYTHONPATH
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

MODEL_SIZE=$4
BATCH_SIZE=$5
SOURCE_SEQ_LEN=$6
TARGET_SEQ_LEN=$7
LR=$8
MIN_LR=$9
PR=${10}
DO=${11}
TRAIN_DATASET_PATH=${12}
VALID_DATASET_PATH=${13}
PRETRAIN_CHECKPOINT_PATH=${14}
EPOCH=${15}
OUTPUT_BASEPATH=${16}

if [ $MODEL_SIZE = 6B ]; then

NUM_LAYERS=28
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
SEQ_LEN=2048

fi

if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16"
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
fi

if [ $DO = true ]; then
    do_options=" \
		    --use-distributed-optimizer"

elif [ $DO = false ]; then
    do_options=" \
                    "
fi

NAME="${ENV}-finetune-huggingface-chatglm-${MODEL_SIZE}-ep-${EPOCH}-lr-${LR}-bs-${BATCH_SIZE}-seqlen-${SEQ_LEN}-pr-${PR}-do-${DO}"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}

FINETUNE_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

megatron_options="  \
        --transformer-type huggingface \
        --load ${PRETRAIN_CHECKPOINT_PATH} \
        --save ${FINETUNE_CHECKPOINT_PATH} \
        --train-data ${TRAIN_DATASET_PATH} \
        --valid-data ${VALID_DATASET_PATH} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --keep-last \
        --micro-batch-size ${BATCH_SIZE} \
        --epochs ${EPOCH} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.01 \
        --num-workers 8\
        --log-interval 1 \
        --eval-interval 100 \
        --eval-iters 10 \
        --save-interval 100000000 \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --source-seq-len ${SOURCE_SEQ_LEN} \
        --target-seq-len ${TARGET_SEQ_LEN} \
        --finetune \
        --no-load-optim \
        --DDP-impl local\
        --patch-tokenizer-type ChatGLMTokenizerFromHF
        "

run_cmd="torchrun $DISTRIBUTED_ARGS finetune_huggingface_chatglm.py
${megatron_options} ${do_options} ${pr_options}"


echo ${run_cmd}
eval ${run_cmd}
set +x
