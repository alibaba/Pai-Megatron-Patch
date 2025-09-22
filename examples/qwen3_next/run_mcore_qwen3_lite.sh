#!/bin/bash
set -e

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATCH_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=${MEGATRON_PATCH_PATH}:${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-250908:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1


NUM_NODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU:-$(python -c "import torch; print(torch.cuda.device_count())")}
[ -z "$MASTER_ADDR" ] && export MASTER_ADDR=localhost
[ -z "$MASTER_PORT" ] && export MASTER_PORT=${MASTER_PORT:-$(shuf -n 1 -i 10000-65535)}

TP=1
PP=1
EP=2
ETP=1
CP=1
MBS=1
GBS=8
SEQ_LEN=2048
DATA_PATH=/mnt/data/datasets/mmap_qwen3_datasets_text_document
PRETRAIN_CHECKPOINT_PATH=/mnt/data/ckpts/huggingface/Qwen3-Next-80B-A3B-Instruct
TENSORBOARD_DIR=/mnt/data/jerry.lp/tensorboard/test_qwen3_next_pretrain
mkdir -p ${TENSORBOARD_DIR}
TRAIN_TOKENS=10000000
WARMUP_TOKENS=10000
TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GBS} / ${SEQ_LEN} ))
LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS}  / ${GBS} / ${SEQ_LEN} ))
LR_DECAY_ITERS=$(( ${TRAIN_TOKENS} /  ${GBS} / ${SEQ_LEN} ))

DISTRIBUTED_ARGS=(
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK
    --nproc_per_node $GPUS_PER_NODE 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

MODEL_ARGS_SMALL=(
    --transformer-impl transformer_engine
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --num-layers 96
    --hidden-size 2048
    --ffn-hidden-size 5120
    --moe-ffn-hidden-size 512
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 2
    --hybrid-attention-ratio 0.125 
    --hybrid-mlp-ratio 0.5 
    --hybrid-override-pattern M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*- 
    --is-hybrid-model 
    --normalization RMSNorm
    --qk-layernorm 
    --norm-epsilon 1e-6
    --swiglu
    --disable-bias-linear
    --use-rotary-position-embeddings
    --rotary-base 10000000
    --rotary-percent 0.25
    --seq-length ${SEQ_LEN}
    --max-position-embeddings ${SEQ_LEN}
    --position-embedding-type rope
    --untie-embeddings-and-output-weights
    --moe-router-load-balancing-type aux_loss
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-router-dtype fp32
    --moe-router-pre-softmax
    --moe-aux-loss-coeff 0.001
    --moe-router-score-function softmax
    --moe-router-topk 10
    --moe-shared-expert-intermediate-size 512 
    --num-experts 512
    --extra-vocab-size 421 
    --patch-tokenizer-type Qwen3Tokenizer
)

TRAINING_ARGS=(
    --use-mcore-models
    --load ${PRETRAIN_CHECKPOINT_PATH}
    --micro-batch-size ${MBS} 
    --global-batch-size ${GBS}
    --train-iters ${TRAIN_ITERS}
    --weight-decay 0.01 
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
    --data-path ${DATA_PATH}
    --split 99,1,0
    --dataset MMAP
    --num-workers 32
    --distributed-timeout-minutes 60
    --exit-duration-in-mins 220
    --no-save-optim
    --manual-gc
    --manual-gc-interval 10
    --no-load-optim
    --no-load-rng
    --save-interval 5000000
    --eval-iters 32
    --eval-interval 20000000
    --tensorboard-dir ${TENSORBOARD_DIR}
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --log-throughput
    --log-interval 1
)

INFRA_ARGS=(
    --enable-experimental
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --expert-model-parallel-size ${EP}
    --context-parallel-size ${CP}
    --expert-tensor-parallel-size ${ETP}
    --use-distributed-optimizer
    --sequence-parallel
    --attention-backend auto
    --recompute-granularity selective
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
    --moe-token-dispatcher-type alltoall


)

cmd="torchrun ${DISTRIBUTED_ARGS[@]} pretrain_qwen3_next.py \
    ${MODEL_ARGS_SMALL[@]} \
    ${TRAINING_ARGS[@]} \
    ${INFRA_ARGS[@]}"

echo $cmd
eval $cmd 2>&1 | tee qwen3_next_lite.log ; exit ${PIPESTATUS[0]}
set +x
