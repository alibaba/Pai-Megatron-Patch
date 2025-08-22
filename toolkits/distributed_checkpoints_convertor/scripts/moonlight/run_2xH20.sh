#!/bin/bash
set -e
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
CONVERTOR_DIR=$( dirname $( dirname ${CURRENT_DIR}))
MEGATRON_PATCH_PATH=$( dirname $( dirname ${CONVERTOR_DIR}))
export PYTHONPATH=${MEGATRON_PATCH_PATH}:${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-250624:${CONVERTOR_DIR}/impl:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true # for PyTorch >= 2.6


NUM_NODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU:-2}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}

MODEL_SIZE=$1 # NOTE: not used
LOAD_DIR=$2
SAVE_DIR=$3
MG2HF=$4
USE_CUDA=$5
PR=$6
HF_DIR=$7

OTHER_ARGS=()
if [ ${MG2HF} = true ]; then
    OTHER_ARGS+=(
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model ${HF_DIR}
        --hf-dir ${HF_DIR}
        --mcore2hf
    )
else
    OTHER_ARGS+=(
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model ${LOAD_DIR}
    )
fi

if [ ${USE_CUDA} = true ]; then
    OTHER_ARGS+=(
        --use-gpu
    )
fi

if [ ${PR} = fp16 ]; then
    OTHER_ARGS+=(
        --fp16
    )
elif [ ${PR} = bf16 ]; then
    OTHER_ARGS+=(
        --bf16
    )
fi

if [ -z ${NUM_NODES} ]; then
    echo "Please Provide WORLD_SIZE"
    exit
fi

if [ -z ${NODE_RANK} ]; then
    echo "Please Provide RANK"
    exit
fi

if [ -z ${MASTER_ADDR} ]; then
    echo "Please Provide MASTER_ADDR"
    exit
fi

if [ -z ${MASTER_PORT} ]; then
    echo "Please Provide MASTER_PORT"
    exit
fi

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 27
    --hidden-size 2048
    --ffn-hidden-size 11264
    --moe-ffn-hidden-size 1408
    --normalization RMSNorm
    --swiglu
    --disable-bias-linear
    --num-attention-heads 16

    --seq-length 1
    --max-position-embeddings 8192
    --attention-backend auto # Can use (flash/fused/unfused/local)
    
    --position-embedding-type rope

    --untie-embeddings-and-output-weights

    --moe-grouped-gemm
    --moe-router-score-function sigmoid
    --moe-token-dispatcher-type alltoall
    --moe-router-topk 6
    --moe-router-group-topk 1
    --moe-router-num-groups 1
    --moe-router-enable-expert-bias
    --moe-layer-freq "'([0]*1+[1]*26)'"
    --moe-shared-expert-intermediate-size 2816
    --num-experts 64

    --kv-lora-rank 512 
    --v-head-dim 128 
    --kv-channels 128
    --multi-latent-attention
    --no-rope-fusion
    --qk-layernorm
)

TRAINING_ARGS=(
    --micro-batch-size 1 
    --global-batch-size 1024
    --train-iters 500000 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
)

if [ -z  ${MODEL_PARALLEL_ARGS} ]; then
    MODEL_PARALLEL_ARGS=(
        --tensor-model-parallel-size 1
        --pipeline-model-parallel-size 1
        --expert-model-parallel-size 2
    )
fi

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --eval-iters 10
)

CONVERT_ARGS=(
    --model-type GPT 
    --load-dir ${LOAD_DIR}
    --save-dir ${SAVE_DIR}
    
    --padded-vocab-size 163840
    --no-load-optim
    --no-load-rng

    --logging-level 1
)

cmd="torchrun ${DISTRIBUTED_ARGS[@]} impl/convert.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${CONVERT_ARGS[@]} \
    ${OTHER_ARGS[@]}"

echo $cmd
eval $cmd