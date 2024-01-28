#!/bin/bash
#sh run_finetune_megatron_mixtral.sh dsw ../.. 0.125B 1 1e-5 1e-6 80 80 0 bf16 1 1 sel false false true false /mnt/llama2-datasets/alpaca_data.json /mnt/llama2-datasets/alpaca_data.json /mnt/mixtral-ckpts/Mixtral-8x7B-v0.1 2 /mnt/output_patch_test
set -e
ENV=$1
MEGATRON_PATCH_PATH=$2
MEGATRON_PATH=${MEGATRON_PATCH_PATH}/Megatron-LM-240126
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
if [ $ENV = dsw ]; then
export CUDA_VISIBLE_DEVICES=0,1
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=2
TOTAL_GPUS=$(($GPUS_PER_NODE*$NNODES))

elif [ $ENV = dlc ]; then

NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}
TOTAL_GPUS=$(($GPUS_PER_NODE*$NNODES))

fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

MODEL_SIZE=$3  #7B
BATCH_SIZE=$4
LR=$5
MIN_LR=$6
SEQ_LEN=$7
PAD_LEN=$8
EXTRA_VOCAB_SIZE=$9
PR=${10}
TP=${11}
PP=${12}
AC=${13}
DO=${14}
FL=${15}
SP=${16}
TE=${17}
TRAIN_DATASET_PATH=${18}
VALID_DATASET_PATH=${19}
PRETRAIN_CHECKPOINT_PATH=${20}
EPOCH=${21}
OUTPUT_BASEPATH=${22}

if [ $MODEL_SIZE = 0.125B ]; then

NUM_LAYERS=2
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
INTERMEDIATE_SIZE=14336
MPE=32768
SLW=4096

gqa_options=" \
		    --group-query-attention \
		    --num-query-groups 8"

elif [ $MODEL_SIZE = 7B ]; then

NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
INTERMEDIATE_SIZE=14336
MPE=32768
SLW=4096

gqa_options=" \
		    --group-query-attention \
		    --num-query-groups 8"

fi

if [ $AC = full ]; then
    activation_checkpoint_options=" \
        --recompute-num-layers 1 \
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
elif [ $PR = fp8 ]; then
    pr_options=" \
        --bf16
        --fp8-hybrid \
        --fp8-amax-compute-algo max \
        --fp8-amax-history-len 1024 \
        --transformer-impl transformer_engine"
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

if [ $TE = true ]; then
    te_options=" \
		    --transformer-impl transformer_engine"

elif [ $TE = false ]; then
    te_options=" \
                    "
fi

if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_options=" \
		    --sequence-parallel"

elif [ $SP = false ]; then
    sp_options=" \
                    "
fi

if [ $PRETRAIN_CHECKPOINT_PATH != none ]; then
    load_options=" \
            --load $PRETRAIN_CHECKPOINT_PATH"
fi

EP=$(($TOTAL_GPUS/$TP/$PP))

FT_NAME="${ENV}-finetune-megatron-llama-${MODEL_SIZE}-lr-${LR}-ep-${EPOCH}-bs-${BATCH_SIZE}-seqlen-${SEQ_LEN}-pr-${PR}--do-${DO}-tp-${TP}-ac-${AC}-sp-${SP}"
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
        --train-data-path ${TRAIN_DATASET_PATH} \
        --valid-data-path ${VALID_DATASET_PATH} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MPE} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
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
        --no-load-optim \
        --no-load-rng \
        --seed 1234 \
        --max-padding-length ${PAD_LEN} \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --patch-tokenizer-type MistralTokenizer \
        --dataset LLama-SFT \
        --swiglu \
        --use-rotary-position-embeddings \
        --position-embedding-type rope \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --normalization RMSNorm \
        --no-masked-softmax-fusion \
        --no-position-embedding \
        --num-experts 8 \
        --moe-router-topk 2 \
        --use-mcore-models \
        --no-rope-fusion \
        --expert-model-parallel-size ${EP} \
        --transformer-impl transformer_engine
        "

run_cmd="torchrun $DISTRIBUTED_ARGS finetune_megatron_mixtral.py
 ${megatron_options} ${pr_options} ${load_options} ${te_options} ${activation_checkpoint_options} ${do_options} ${flash_options} ${sp_options} ${gqa_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
