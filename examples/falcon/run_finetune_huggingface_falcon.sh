#!/bin/bash
#sh run_finetune_huggingface_falcon.sh dsw /workspace/Megatron-LM/ /workspace/PAI-Megatron-Patch/ 7B 1 1e-5 1e-6 2048 80 1 fp16 true /mnt/llama-datasets/alpaca_data.json /mnt/llama-datasets/alpaca_data.json /mnt/falcon-ckpts/falcon-7b-hf/ 2 /mnt/output_falcon
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
LR=$6
MIN_LR=$7
SEQ_LEN=$8
PAD_LEN=$9
EXTRA_VOCAB_SIZE=${10}
PR=${11}
DO=${12}
TRAIN_DATASET_PATH=${13}
VALID_DATASET_PATH=${14}
PRETRAIN_CHECKPOINT_PATH=${15}
EPOCH=${16}
OUTPUT_BASEPATH=${17}

if [ $MODEL_SIZE = 7B ]; then

NUM_LAYERS=32
HIDDEN_SIZE=4544
NUM_ATTN_HEADS=71

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

NAME="${ENV}-finetune-huggingface-falcon-${MODEL_SIZE}-ep-${EPOCH}-lr-${LR}-bs-${BATCH_SIZE}-seqlen-${SEQ_LEN}-pr-${PR}-do-${DO}"
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
        --num-workers 0\
        --log-interval 1 \
        --eval-interval 100 \
        --eval-iters 10 \
        --save-interval 100000000 \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --finetune \
        --max-padding-length ${PAD_LEN} \
        --no-load-optim \
        --DDP-impl local\
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --patch-tokenizer-type FalconTokenizer
        "

run_cmd="torchrun $DISTRIBUTED_ARGS finetune_huggingface_falcon.py
${megatron_options} ${do_options} ${pr_options}"


echo ${run_cmd}
eval ${run_cmd}
set +x
