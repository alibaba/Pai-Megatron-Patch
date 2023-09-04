#!/bin/bash
#sh run_pretrain_megatron_baichuan.sh dsw /workspace/Megatron-LM/ /workspace/PAI-Megatron-Patch/ 7B 1 8 1e-5 1e-6 2048 80 1 fp16 1 1 sel true false false 100000 /mnt/baichuan-datasets/alpaca_data.json /mnt/baichuan-ckpts/baichuan-13b-hf-to-megatron-tp1-pp1 100000000 10000 /mnt/output_baichuan
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
GLOBAL_BATCH_SIZE=$6
LR=$7
MIN_LR=$8
SEQ_LEN=$9
PAD_LEN=${10}
EXTRA_VOCAB_SIZE=${11}
PR=${12}
TP=${13}
PP=${14}
AC=${15}
DO=${16}
FL=${17}
SP=${18}
TE=${19}
SAVE_INTERVAL=${20}
DATASET_PATH=${21}
PRETRAIN_CHECKPOINT_PATH=${22}
TRAIN_TOKENS=${23}
WARMUP_TOKENS=${24}
OUTPUT_BASEPATH=${25}


if [ $MODEL_SIZE = 7B ]; then

NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
INTERMEDIATE_SIZE=11008

elif [ $MODEL_SIZE = 13B ]; then

NUM_LAYERS=40
HIDDEN_SIZE=5120
NUM_ATTN_HEADS=40
INTERMEDIATE_SIZE=13696

fi

if [ $PRETRAIN_CHECKPOINT_PATH != none ]; then
    load_options=" \
		    --load $PRETRAIN_CHECKPOINT_PATH"
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

TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS}  / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_DECAY_ITERS=$(( ${TRAIN_TOKENS} /  ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))

NAME="${ENV}-pretrain-megatron-baichuan-${MODEL_SIZE}-lr-${LR}-bs-${BATCH_SIZE}-seqlen-${SEQ_LEN}-pr-${PR}-tp-${TP}-pp-${PP}-ac-${AC}-do-${DO}-sp-${SP}-tt-${TRAIN_TOKENS}-wt-${WARMUP_TOKENS}"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}

SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

megatron_options="  \
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --split 98,2,0 \
        --data-impl mmap \
        --data-path ${DATASET_PATH}
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style linear \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --init-method-std 0.006 \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --intermediate-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --log-interval 1 \
        --eval-interval 100 \
        --eval-iters 10 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --DDP-impl local \
        --no-load-optim \
        --no-load-rng \
        --num-workers 8 \
        --seed 1234 \
        --dataset LLama-SFT \
        --max-padding-length ${PAD_LEN} \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --patch-tokenizer-type BaichuanTokenizer \
        --swiglu \
        --use-alibi-mask \
        --position-embedding-type none \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear
        "

run_cmd="torchrun $DISTRIBUTED_ARGS pretrain_megatron_baichuan13b.py
 ${megatron_options} ${activation_checkpoint_options} ${do_options} ${pr_options} ${sp_options} ${flash_options} ${load_options} ${te_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
