#!/bin/bash
# sh run_evaluate_megatron_glm130b.sh dsw /workspace/Megatron-LM/ /workspace/SwissArmyTransformer/ /workspace/PAI-Megatron-Patch/ 2B 1 256 128 fp16 1 1 /mnt/wikitext-103/wiki.test.tokens none
set -e
ENV=$1
MEGATRON_PATH=$2
SAT_PATH=$3
MEGATRON_PATCH_PATH=$4
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:${SAT_PATH}:$PYTHONPATH
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

MODEL_SIZE=$5
BATCH_SIZE=$6
SEQ_LEN=$7
GEN_LEN=$8
PR=$9
TP=${10}
PP=${11}
DATASET_PATH=${12}
PRETRAIN_CHECKPOINT_PATH=${13}

if [ $MODEL_SIZE = 2B ]; then

NUM_LAYERS=6
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=32


elif [ $MODEL_SIZE = 10B ]; then

NUM_LAYERS=48
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=64


elif [ $MODEL_SIZE = 130B ]; then

NUM_LAYERS=70
HIDDEN_SIZE=12288
NUM_ATTN_HEADS=96

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


megatron_options=" \
        --data-path ${DATASET_PATH}
        --micro-batch-size ${BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --log-interval 1 \
        --eval-interval 100 \
        --eval-iters 10 \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --DDP-impl local \
        --no-load-optim \
        --no-load-rng \
        --seed 1234 \
        --num-workers 0 \
        --dataset GLM130B-WIKITEXT103 \
        --use-distributed-optimizer \
        --generation-length ${GEN_LEN} \
        --position-embedding-type rotary \
        --apply-residual-connection-post-layernorm \
        --glu-activation geglu \
        --patch-tokenizer-type IcetkGLM130BTokenizer
        "

run_cmd="torchrun $DISTRIBUTED_ARGS evaluate_sat_glm130b.py
 ${megatron_options} ${activation_checkpoint_options} ${load_options} ${pr_options} ${sp_options} ${flash_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
