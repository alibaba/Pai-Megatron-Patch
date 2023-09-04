#!/bin/bash
#sh run_finetune_megatron_starcoder.sh dsw /workspace/Megatron-LM/ /workspace/PAI-Megatron-Patch/ 16B 8 1e-5 1e-6 2048 512 0 fp16 4 1 sel true false false  /mnt/starcoder-datasets/alpaca_data.json /mnt/alpaca-datasets/alpaca_data.json /mnt/alpaca-ckpts/starcoder-16b-hf-to-megatron-tp4-pp1 5 /mnt/output_starcoder
set -e
ENV=$1
MEGATRON_PATH=$2
MEGATRON_PATCH_PATH=$3
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
if [ $ENV = dsw ]; then
export CUDA_VISIBLE_DEVICES=0,1,2,3
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=4

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
TP=${12}
PP=${13}
AC=${14}
DO=${15}
FL=${16}
SP=${17}
TRAIN_DATASET_PATH=${18}
VALID_DATASET_PATH=${19}
PRETRAIN_CHECKPOINT_PATH=${20}
EPOCH=${21}
OUTPUT_BASEPATH=${22}


if [ $MODEL_SIZE = 16B ]; then

NUM_LAYERS=40
HIDDEN_SIZE=6144
NUM_ATTN_HEADS=48
INTERMEDIATE_SIZE=24576

elif [ $MODEL_SIZE = 7B ]; then

NUM_LAYERS=42
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
INTERMEDIATE_SIZE=16384

elif [ $MODEL_SIZE = 3B ]; then

NUM_LAYERS=36
HIDDEN_SIZE=2816
NUM_ATTN_HEADS=22
INTERMEDIATE_SIZE=11264

elif [ $MODEL_SIZE = 1B ]; then

NUM_LAYERS=24
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
INTERMEDIATE_SIZE=8192

fi

if [ $AC = full ]; then
    activation_checkpoint_options=" \
		    --recompute-method uniform \
		    --recompute-granularity full"
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-activations"
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
                    "
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

if [ $FL = true ]; then
    flash_options=" \
		    --use-flash-attn"

elif [ $FL = false ]; then
    flash_options=" \
                    "
fi

if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_options=" \
		    --sequence-parallel"

elif [ $SP = false ]; then
    sp_options=" \
                    "
fi

FT_NAME="${ENV}-finetune-megatron-starcoder-${MODEL_SIZE}-lr-${LR}-ep-${EPOCH}-bs-${BATCH_SIZE}-seqlen-${SEQ_LEN}-pr-${PR}--do-${DO}-tp-${TP}-ac-${AC}-sp-${SP}"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${FT_NAME}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}

FINETUNE_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${FT_NAME}"

megatron_options="  \
        --load ${PRETRAIN_CHECKPOINT_PATH} \
        --save ${FINETUNE_CHECKPOINT_PATH} \
        --train-data ${TRAIN_DATASET_PATH} \
        --valid-data ${VALID_DATASET_PATH} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings 8192  \
        --intermediate-size ${INTERMEDIATE_SIZE} \
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
        --eval-interval 1000 \
        --eval-iters 10 \
        --save-interval 1000000 \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --finetune \
        --DDP-impl local \
        --no-load-optim \
        --no-load-rng \
        --seed 1234 \
        --max-padding-length ${PAD_LEN} \
        --attention-head-type multiquery \
        --patch-tokenizer-type StarcoderTokenizerFromHF
        "

run_cmd="torchrun $DISTRIBUTED_ARGS finetune_megatron_starcoder.py
 ${megatron_options} ${activation_checkpoint_options} ${do_options} ${pr_options} ${sp_options} ${flash_options}"


echo ${run_cmd}
eval ${run_cmd}
set +x
