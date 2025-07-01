#!/bin/bash
set -e

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATCH_PATH=$( dirname $(dirname $( dirname ${CURRENT_DIR})))
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATCH_PATH}:${MEGATRON_PATCH_PATH}/backends/megatron/PAI-Megatron-LM-240718

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true # for PyTorch >= 2.6

ENV=$1
if [ $ENV = dsw ]; then
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=`python -c "import torch; print(torch.cuda.device_count())"`

elif [ $ENV = dlc ]; then

NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}

fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

MODEL_SIZE=$2
BATCH_SIZE=$3
GLOBAL_BATCH_SIZE=$4
LR=$5
MIN_LR=$6
SEQ_LEN=$7
PAD_LEN=$8
PR=$9
TP=${10}
PP=${11}
EP=${12}
AC=${13}
DO=${14}
FL=${15}
SP=${16}
TE=${17}
SAVE_INTERVAL=${18}
DATASET_PATH=${19}
PRETRAIN_CHECKPOINT_PATH=${20}
TRAIN_TOKENS=${21}
WARMUP_TOKENS=${22}
OUTPUT_BASEPATH=${23}

if [ $MODEL_SIZE = 0.5B ]; then

HIDDEN_SIZE=896
INTERMEDIATE_SIZE=4864
MAX_POSITION_EMBEDDINGS=131072
MAX_WINDOW_LAYERS=24
NUM_ATTENTION_HEADS=14
NUM_HIDDEN_LAYERS=24
NUM_KEY_VALUE_HEADS=2
RMS_NORM_EPS=1e-6
ROPE_THETA=1000000
SLIDING_WINDOW=131072
EXTRA_VOCAB_SIZE=34

moe_options=" \
            "

elif [ $MODEL_SIZE = 1.5B ]; then

HIDDEN_SIZE=1536
INTERMEDIATE_SIZE=8960
MAX_POSITION_EMBEDDINGS=131072
MAX_WINDOW_LAYERS=28
NUM_ATTENTION_HEADS=12
NUM_HIDDEN_LAYERS=28
NUM_KEY_VALUE_HEADS=2
RMS_NORM_EPS=1e-6
ROPE_THETA=1000000
SLIDING_WINDOW=131072
EXTRA_VOCAB_SIZE=293

moe_options=" \
            "

elif [ $MODEL_SIZE = 7B ]; then

HIDDEN_SIZE=3584
INTERMEDIATE_SIZE=18944
MAX_POSITION_EMBEDDINGS=131072
MAX_WINDOW_LAYERS=28
NUM_ATTENTION_HEADS=28
NUM_HIDDEN_LAYERS=28
NUM_KEY_VALUE_HEADS=4
RMS_NORM_EPS=1e-6
ROPE_THETA=1000000
SLIDING_WINDOW=131072
EXTRA_VOCAB_SIZE=421

moe_options=" \
            "

elif [ $MODEL_SIZE = 70B ]; then

HIDDEN_SIZE=8192
INTERMEDIATE_SIZE=28672
MAX_POSITION_EMBEDDINGS=4096
MAX_WINDOW_LAYERS=80
NUM_ATTENTION_HEADS=64
NUM_HIDDEN_LAYERS=80
NUM_KEY_VALUE_HEADS=8
RMS_NORM_EPS=1e-5
ROPE_THETA=1000000
SLIDING_WINDOW=131072
EXTRA_VOCAB_SIZE=0

moe_options=" \
            "

elif [ $MODEL_SIZE = 72B ]; then

HIDDEN_SIZE=8192
INTERMEDIATE_SIZE=29568
MAX_POSITION_EMBEDDINGS=131072
MAX_WINDOW_LAYERS=80
NUM_ATTENTION_HEADS=64
NUM_HIDDEN_LAYERS=80
NUM_KEY_VALUE_HEADS=8
RMS_NORM_EPS=1e-5
ROPE_THETA=1000000
SLIDING_WINDOW=131072
EXTRA_VOCAB_SIZE=421

moe_options=" \
            "

elif [ $MODEL_SIZE = A14B ]; then

HIDDEN_SIZE=3584
INTERMEDIATE_SIZE=18944
MAX_POSITION_EMBEDDINGS=131072
MAX_WINDOW_LAYERS=28
MOE_INTERMEDIATE_SIZE=2560
NUM_ATTENTION_HEADS=28
NUM_EXPERTS=64
NUM_EXPERTS_PER_TOPK=8
NUM_HIDDEN_LAYERS=28
NUM_KEY_VALUE_HEADS=4
RMS_NORM_EPS=1e-6
ROPE_THETA=1000000
SHARED_EXPERT_INTERMEDIATE_SIZE=20480
SLIDING_WINDOW=131072
EXTRA_VOCAB_SIZE=293

moe_options=" \
            --moe-router-topk ${NUM_EXPERTS_PER_TOPK} \
            --num-experts ${NUM_EXPERTS} \
            --expert-model-parallel-size ${EP}\
            --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
            --shared-moe-ffn-hidden-size ${SHARED_EXPERT_INTERMEDIATE_SIZE} \
            --enable-shared-expert"

fi

if [ $AC = full ]; then
    activation_checkpoint_options=" \
		    --recompute-method uniform \
		    --recompute-num-layers 1 \
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
		    --fp16 \
        --apply-query-key-layer-scaling"
    export NVTE_APPLY_QK_LAYER_SCALING=1
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
elif [ $PR = fp8 ]; then
    pr_options=" \
        --bf16 \
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
        --transformer-impl local"
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

TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS}  / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_DECAY_ITERS=$(( ${TRAIN_TOKENS} /  ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))

NAME="pretrain-mcore-llama3-${MODEL_SIZE}-lr-${LR}-minlr-${MIN_LR}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}-pr-${PR}-tp-${TP}-pp-${PP}-ac-${AC}-do-${DO}-sp-${SP}-tt-${TRAIN_TOKENS}-wt-${WARMUP_TOKENS}"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}

SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

megatron_options="  \
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --data-path ${DATASET_PATH} \
        --split 99,1,0 \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 0.0 \
        --init-method-std 0.008 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_HIDDEN_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTENTION_HEADS} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --max-padding-length ${PAD_LEN} \
        --log-interval 1 \
        --eval-interval 10000 \
        --eval-iters 10 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --no-load-optim \
        --no-load-rng \
        --num-workers 8 \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --patch-tokenizer-type LLama3Tokenizer \
        --dataset LLama-Pretrain-Idxmap \
        --swiglu \
        --normalization RMSNorm \
        --norm-epsilon ${RMS_NORM_EPS} \
        --use-rotary-position-embeddings \
        --no-rope-fusion \
        --position-embedding-type rope \
        --untie-embeddings-and-output-weights \
        --no-log-loss-scale-to-tensorboard \
        --disable-bias-linear \
        --group-query-attention \
        --num-query-groups ${NUM_KEY_VALUE_HEADS} \
        --rotary-percent 1.0 \
        --rotary-base ${ROPE_THETA} \
        --rotary-seq-len-interpolation-factor 1 \
        --optimizer cpu-adam
        "

run_cmd="torchrun $DISTRIBUTED_ARGS pretrain_llama_mcore070.py
 ${megatron_options} ${pr_options} ${load_options} ${te_options} ${activation_checkpoint_options} ${do_options} ${flash_options} ${sp_options} ${moe_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
