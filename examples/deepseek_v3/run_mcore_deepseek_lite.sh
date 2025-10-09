#!/bin/bash
set -e

export NVSHMEM_IB_GID_INDEX=3
export ACCL_LOAD_BALANCE=1
export ACCL_TOPO_FIX=1
export ACCL_EP_ENV_CHECK=1
export ACCL_NORMAL_MODE=IBGDA
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATCH_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=${MEGATRON_PATCH_PATH}:${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-250624:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true # for PyTorch >= 2.6


NUM_NODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU:-$(python -c "import torch; print(torch.cuda.device_count())")}
[ -z "$MASTER_ADDR" ] && export MASTER_ADDR=localhost
[ -z "$MASTER_PORT" ] && export MASTER_PORT=${MASTER_PORT:-$(shuf -n 1 -i 10000-65535)}

TP=2
PP=8
EP=16
CP=1
MBS=1
GBS=8192
SEQ_LEN=4096
DATA_PATH='1.0 /mnt/data/datasets/mmap_deepseekv2_datasets_text_document'
EXTRA_VOCAB_SIZE=467
PRETRAIN_CHECKPOINT_PATH=/mnt/data/ckpts/huggingface/DeepSeek-R1-BF16
TRAIN_TOKENS=1000000000
WARMUP_TOKENS=100000
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

MODEL_ARGS=(
    --transformer-impl transformer_engine
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --num-layers 61
    --hidden-size 7168
    --ffn-hidden-size 18432
    --moe-ffn-hidden-size 2048
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --swiglu
    --disable-bias-linear
    --num-attention-heads 128
    --seq-length ${SEQ_LEN}
    --max-position-embeddings ${SEQ_LEN}
    --position-embedding-type rope
    --untie-embeddings-and-output-weights
    --moe-router-load-balancing-type seq_aux_loss
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-router-dtype fp32
    --moe-router-pre-softmax
    --moe-aux-loss-coeff 1e-4
    --moe-router-topk-scaling-factor 2.5
    --moe-router-score-function sigmoid
    --moe-router-topk 8
    --moe-router-group-topk 4
    --moe-router-num-groups 8
    --moe-router-enable-expert-bias
    --moe-layer-freq "'([0]*3+[1]*58)'"
    --moe-shared-expert-intermediate-size 2048 
    --num-experts 256
    --q-lora-rank 1536 
    --kv-lora-rank 512 
    --v-head-dim 128 
    --rotary-scaling-factor 40
    --mscale 1.0
    --mscale-all-dim 1.0
    --multi-latent-attention
    --qk-layernorm
    --moe-router-force-load-balancing
    --extra-vocab-size ${EXTRA_VOCAB_SIZE} 
    --patch-tokenizer-type DeepSeekV2Tokenizer
)

MODEL_ARGS_SMALL=(
    --transformer-impl transformer_engine
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --num-layers 61
    --hidden-size 64
    --ffn-hidden-size 128
    --moe-ffn-hidden-size 256
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --swiglu
    --disable-bias-linear
    --num-attention-heads 8
    --seq-length ${SEQ_LEN}
    --max-position-embeddings ${SEQ_LEN}
    --position-embedding-type rope
    --untie-embeddings-and-output-weights
    --moe-router-load-balancing-type seq_aux_loss
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-router-dtype fp32
    --moe-router-pre-softmax
    --moe-aux-loss-coeff 1e-4
    --moe-router-topk-scaling-factor 2.5
    --moe-router-score-function sigmoid
    --moe-router-topk 8
    --moe-router-group-topk 4
    --moe-router-num-groups 8
    --moe-router-enable-expert-bias
    --moe-layer-freq "'([0]*3+[1]*58)'"
    --moe-shared-expert-intermediate-size 2048 
    --num-experts 32
    --q-lora-rank 1536 
    --kv-lora-rank 512 
    --v-head-dim 128 
    --rotary-scaling-factor 40
    --mscale 1.0
    --mscale-all-dim 1.0
    --multi-latent-attention
    --qk-layernorm
    --moe-router-force-load-balancing
    --extra-vocab-size ${EXTRA_VOCAB_SIZE} 
    --patch-tokenizer-type DeepSeekV2Tokenizer
)

TRAINING_ARGS=(
    --use-mcore-models
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
    --data-path ${DATA_PATH}
    --split 99,1,0
    --dataset MMAP
    --num-workers 6
    --distributed-timeout-minutes 60
    --exit-duration-in-mins 220
    --no-save-optim
    --no-check-for-nan-in-loss-and-grad
    --manual-gc
    --manual-gc-interval 10
    --no-load-optim
    --no-load-rng
    --auto-detect-ckpt-format
    --save-interval 5000000
    --eval-iters 32
    --eval-interval 20000000
    --dist-ckpt-strictness log_all
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
    --expert-tensor-parallel-size 1
    --use-distributed-optimizer
    --sequence-parallel
    --attention-backend auto
    --recompute-granularity selective
    --recompute-modules mla_up_proj
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
    --moe-token-dispatcher-type flex
    --pipeline-model-parallel-layout 'Et\|\(tt\|\)*30L'
    --moe-enable-deepep
    --fp8-format hybrid 
    --fp8-recipe blockwise

)
#    --fp8-format hybrid --fp8-recipe blockwise
#    --decoder-first-pipeline-num-layers 8  --decoder-last-pipeline-num-layers 5 --pipeline-model-parallel-layout 'Et*3\|\(tt\|\)*29\|L'
#    --moe-enable-deepep, --moe-token-dispatcher-type flex
cmd="torchrun ${DISTRIBUTED_ARGS[@]} pretrain_deepseek_250624.py \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${INFRA_ARGS[@]}"

echo $cmd
eval $cmd 2>&1 | tee log_new.txt ; exit ${PIPESTATUS[0]}
set +x
