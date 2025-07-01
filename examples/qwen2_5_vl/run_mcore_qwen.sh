#!/bin/bash
set -e
ENV=$1
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATCH_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=${MEGATRON_PATCH_PATH}:${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-250217:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true # for PyTorch >= 2.6

if [ -z ${MP_AC_LAYERS} ];then
    MP_AC_LAYERS=1
fi

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

if [ -z ${MP_VP} ]; then
    vp_options=""
else
    vp_options=" \
        --num-layers-per-virtual-pipeline-stage ${MP_VP}"
fi

MP_SFT_PACKING=false

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

### BASE CONFIG ###
MODEL_SIZE=$2
BATCH_SIZE=$3
GLOBAL_BATCH_SIZE=$4
LR=$5
MIN_LR=$6
SEQ_LEN=$7
PAD_LEN=$8
PR=$9
### BASE CONFIG ###

### PARALLEL / BOOL OPTION ###
TP=${10}
PP=${11}
CP=${12}
SP=${13}
DO=${14}
FL=${15}
### PARALLEL / BOOL OPTION ###

### OTHERS ###
AC=${16}
OPTIMIZER_OFFLOAD=${17}
SAVE_INTERVAL=${18}
DATASET_PATH=${19}
VALID_DATASET_PATH=${20}
PRETRAIN_CHECKPOINT_PATH=${21}

TRAIN_ITERS=${22}
LR_WARMUP_ITERS=${23}
###############################

OUTPUT_BASEPATH=${24}
### OTHERS ###
if [ $FL = true ]; then
    export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0
elif [ $FL = false ]; then
    export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1
fi

if [ $MODEL_SIZE = 3B ]; then

NUM_LAYERS=36 # num_hidden_layers
HIDDEN_SIZE=2048 # hidden_size
NUM_ATTN_HEADS=16 # num_attention_heads
INTERMEDIATE_SIZE=11008 # intermediate_size
NUM_KEY_VALUE_HEADS=2 # num_key_value_heads
MAX_POSITION_EMBEDDINGS=128000 # max_position_embeddings
EXTRA_VOCAB_SIZE=293 # 151643 + 293 = 151936
RMS_NORM_EPS=1e-6 # rms_norm_eps


gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"


tie_option="" # tie_word_embeddings


elif [ $MODEL_SIZE = 7B ]; then

NUM_LAYERS=28
HIDDEN_SIZE=3584
NUM_ATTN_HEADS=28
INTERMEDIATE_SIZE=18944
NUM_KEY_VALUE_HEADS=4
MAX_POSITION_EMBEDDINGS=128000
EXTRA_VOCAB_SIZE=421  # 151643 + 421 = 152064
RMS_NORM_EPS=1e-6

gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"


tie_option=" \
        --untie-embeddings-and-output-weights \
        "

elif [ $MODEL_SIZE = 32B ]; then

NUM_LAYERS=64
HIDDEN_SIZE=5120
NUM_ATTN_HEADS=40
INTERMEDIATE_SIZE=27648
NUM_KEY_VALUE_HEADS=8
MAX_POSITION_EMBEDDINGS=128000
EXTRA_VOCAB_SIZE=421  # 151643 + 421 = 152064
RMS_NORM_EPS=1e-6

gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"


tie_option=" \
        --untie-embeddings-and-output-weights \
        "

elif [ $MODEL_SIZE = 72B ]; then

NUM_LAYERS=80
HIDDEN_SIZE=8192
NUM_ATTN_HEADS=64
INTERMEDIATE_SIZE=29568
NUM_KEY_VALUE_HEADS=8
MAX_POSITION_EMBEDDINGS=128000
EXTRA_VOCAB_SIZE=421
RMS_NORM_EPS=1e-5

gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"


tie_option=" \
        --untie-embeddings-and-output-weights \
        "


fi

if [ -z ${MP_PP0_LAYERS} ];then
    uneven_split_option=""
elif [ ${PP} -gt 1 ]; then
    _check=$(( ( $NUM_LAYERS - ${MP_PP0_LAYERS} ) % ( ${PP} - 1 ) ))
    if [ $_check != 0 ]; then
        echo "With uneven pipelineing the left over layers must be divisible by left over stages."
        exit -1
    fi

    uneven_split_option=" \
        --decoder-first-pipeline-num-layers ${MP_PP0_LAYERS}
    "
else
    echo "uneven pipeline split must be used when PP > 1"
    exit -1
fi

if [ $AC = full ]; then
    _check=$(( ($NUM_LAYERS / $PP) % ${MP_AC_LAYERS} ))
    if [ $_check != 0 ]; then
        echo "the num layers per pp rank must be a multiple of the recompute layers."
        exit -1
    fi
    activation_checkpoint_options=" \
		    --recompute-method uniform \
            --recompute-num-layers ${MP_AC_LAYERS} \
		    --recompute-granularity full"
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-activations"
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
    "
elif [ $AC = offload ]; then
    activation_checkpoint_options=" \
		    --cpu-offloading \
		    --cpu-offloading-num-layers ${MP_AC_LAYERS}"
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
        --fp8-format hybrid \
        --fp8-amax-compute-algo max \
        --fp8-amax-history-len 1024"
fi

if [ $OPTIMIZER_OFFLOAD != false ] && [ $DO = false ]; then
    echo "Offload optimizer is valid only if \$DO=true"
    DO=true
fi

if [ $DO = true ]; then
    do_options=" \
		    --use-distributed-optimizer"

elif [ $DO = false ]; then
    do_options=" \
                    "
fi

comm_overlap_option="\
        --overlap-grad-reduce \
        --overlap-param-gather"

if [ $SP = true ] && [ $TP -gt 1 ]; then
    comm_overlap_option="\
        --tp-comm-overlap \
        --overlap-grad-reduce \
        --overlap-param-gather \
        --sequence-parallel"
fi

if [ $PRETRAIN_CHECKPOINT_PATH != none ]; then
    load_options=" \
            --load $PRETRAIN_CHECKPOINT_PATH"
fi

LR_DECAY_ITERS=$(( ${TRAIN_ITERS} - ${LR_WARMUP_ITERS}))
PREFIX="finetune-mcore-qwen2-vl-${MODEL_SIZE}-lr-${LR}-minlr-${MIN_LR}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}"

if [ ${MP_SFT_PACKING} = true ]; then
    packing_options=" \
        --reset-position-ids \
        --no-create-attention-mask-in-dataloader
    "
else
    packing_options=""
fi

##### Prepare logdirs #######
NAME="${PREFIX}-pr-${PR}-tp-${TP}-pp-${PP}-cp-${CP}-ac-${AC}-do-${DO}-sp-${SP}-ti-${TRAIN_ITERS}-wi-${LR_WARMUP_ITERS}"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}
SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

mkdir -p ${SAVED_PRETRAIN_CHECKPOINT_PATH}
find -L ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${SAVED_PRETRAIN_CHECKPOINT_PATH}
find -L ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "merges.txt" -print0 | xargs -0 cp -t ${SAVED_PRETRAIN_CHECKPOINT_PATH}

megatron_options="  \
        --train-data-path ${DATASET_PATH} \
        --valid-data-path ${VALID_DATASET_PATH} \
        --split 100,0,0 \
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --weight-decay 0.01 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        --init-method-std 0.02 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --max-padding-length ${PAD_LEN} \
        --log-interval 1 \
        --log-throughput \
        --eval-interval 1000 \
        --eval-iters 10 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --context-parallel-size ${CP} \
        --no-load-optim \
        --no-load-rng \
        --num-workers 8 \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --patch-tokenizer-type Qwen2VLTokenizer \
        --swiglu \
        --normalization RMSNorm \
        --norm-epsilon ${RMS_NORM_EPS} \
        --use-rotary-position-embeddings \
        --position-embedding-type rope \
        --disable-bias-linear \
        --add-qkv-bias \
        --rotary-percent 1.0 \
        --rotary-base 1000000 \
        --rotary-seq-len-interpolation-factor 1 \
        --no-save-optim \
        --disable-vision-class-token \
        --dataloader-type external \
        --transformer-impl transformer_engine \
        --ckpt-format torch \
        "

run_cmd="torchrun $DISTRIBUTED_ARGS pretrain_qwen.py
 ${megatron_options} ${dataset_option} ${pr_options} ${load_options} ${activation_checkpoint_options} \
 ${do_options} ${gqa_options} ${sft_option} ${tie_option} ${packing_options} ${uneven_split_option} ${vp_options} ${comm_overlap_option}"

echo ${run_cmd}
eval ${run_cmd}
set +x
