#!/bin/bash
set -e

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATCH_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=${MEGATRON_PATCH_PATH}:${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-250624:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1
DISTRIBUTED_ARGS=(
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --nproc_per_node $GPUS_PER_NODE
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

TP=1
PP=1
EP=1
ETP=1
CP=1
MBS=1
GBS=32
SEQ_LEN=4096
TRAIN_DATA_PATH=${your_wds_output_dir}
VALID_DATA_PATH=${your_wds_output_dir}
PRETRAIN_CHECKPOINT_PATH=/mnt/qwen3-vl-ckpts/Qwen3-VL-30B-A3B-Instruct
TRAIN_ITERS=500
LR_WARMUP_ITERS=50
LR_DECAY_ITERS=450

MODEL_ARGS_SMALL=(
    --transformer-impl transformer_engine
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --num-layers 4
    --hidden-size 2048
    --ffn-hidden-size 6144
    --moe-ffn-hidden-size 768
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --swiglu
    --disable-bias-linear
    --num-attention-heads 32
    --seq-length ${SEQ_LEN}
    --max-position-embeddings 262144
    --max-padding-length ${SEQ_LEN}
    --position-embedding-type rope
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 4
    --moe-router-load-balancing-type aux_loss
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --moe-permute-fusion
    --moe-router-dtype fp32
    --moe-aux-loss-coeff 0.001
    --moe-router-score-function sigmoid
    --moe-router-topk 8
    --moe-layer-freq "'([1]*4)'"
    --num-experts 128
    --mrope-section 24 20 20
    --patch-size 16
    --qk-layernorm
    --kv-channels 128
    --use-rotary-position-embeddings
    --position-embedding-type mrope
    --rotary-base 1000000
    --rotary-seq-len-interpolation-factor 1
    --rotary-percent 1.0
    --padded-vocab-size 151936
    --patch-tokenizer-type Qwen2VLTokenizer
)

TRAINING_ARGS=(
    --load ${PRETRAIN_CHECKPOINT_PATH}
    --micro-batch-size ${MBS} 
    --global-batch-size ${GBS}
    --train-iters ${TRAIN_ITERS}
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-decay-iters ${LR_DECAY_ITERS}
    --lr-warmup-iters ${LR_WARMUP_ITERS}
    --train-data-path ${TRAIN_DATA_PATH}
    --valid-data-path ${VALID_DATA_PATH}
    --split 99,1,0
    --num-workers 0
    --disable-vision-class-token
    --dataloader-type external
    --distributed-timeout-minutes 60
    --exit-duration-in-mins 220
    --no-save-optim
    --no-load-optim
    --no-load-rng
    --save-interval 5000000
    --eval-iters 32
    --eval-interval 20000000
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --log-throughput
    --log-interval 1
)

INFRA_ARGS=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --expert-model-parallel-size ${EP}
    --context-parallel-size ${CP}
    --expert-tensor-parallel-size ${ETP}
    --use-distributed-optimizer
    --sequence-parallel
    --attention-backend flash
    --recompute-granularity selective
    --overlap-grad-reduce
    --overlap-param-gather
)

cmd="torchrun ${DISTRIBUTED_ARGS[@]} pretrain_qwen.py \
    ${MODEL_ARGS_SMALL[@]} \
    ${TRAINING_ARGS[@]} \
    ${INFRA_ARGS[@]}"

echo $cmd
eval $cmd 2>&1 | tee test.log ; exit ${PIPESTATUS[0]}
set +x
