#!/bin/bash
#sh run_evaluate_megatron_deepseek.sh dsw /workspace/Pai-Megatron-Patch 7B 1 80 80 0 bf16 2 1 sel true true true false /mnt/deepseek-datasets/alpaca_data.json /mnt/deepseek-ckpts/Llama-2-7b-hf-to-mg-tp2-pp1/
#sh run_evaluate_megatron_deepseek.sh dsw /workspace/Pai-Megatron-Patch 13B 1 80 80 0 bf16 2 1 sel true true true false /mnt/deepseek-datasets/alpaca_data.json /mnt/deepseek-ckpts/Llama-2-13b-hf-to-mg-tp2-pp1/
#sh run_evaluate_megatron_deepseek.sh dsw /workspace/Pai-Megatron-Patch 7B 1 80 80 0 bf16 2 1 sel true false true true /mnt/deepseek-datasets/alpaca_data.json /mnt/deepseek-ckpts/Llama-2-7b-hf-to-te-tp2-pp1/
set -e
ENV=$1
MEGATRON_PATCH_PATH=$2
MEGATRON_PATH=${MEGATRON_PATCH_PATH}/Megatron-LM-231007
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

MODEL_SIZE=$3
BATCH_SIZE=$4
SEQ_LEN=$5
PAD_LEN=$6
EXTRA_VOCAB_SIZE=$7
PR=$8
TP=$9
PP=${10}
AC=${11}
DO=${12}
FL=${13}
SP=${14}
TE=${15}
DATASET_PATH=${16}
PRETRAIN_CHECKPOINT_PATH=${17}

if [ $MODEL_SIZE = 6.7B ]; then

NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
INTERMEDIATE_SIZE=11008

gqa_options=""

elif [ $MODEL_SIZE = 33B ]; then

NUM_LAYERS=62
HIDDEN_SIZE=7168
NUM_ATTN_HEADS=56
INTERMEDIATE_SIZE=19200

gqa_options=" \
                    --group-query-attention \
                    --num-query-groups 8"

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

if [ $PRETRAIN_CHECKPOINT_PATH != none ]; then
    load_options=" \
            --load $PRETRAIN_CHECKPOINT_PATH"
fi


megatron_options=" \
        --valid-data-path ${DATASET_PATH}
        --micro-batch-size ${BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --log-interval 1 \
        --eval-interval 100 \
        --eval-iters 10 \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --no-load-optim \
        --no-load-rng \
        --seed 1234 \
        --num-workers 0 \
        --dataset LLama-Pretrain-Raw \
        --max-padding-length ${PAD_LEN} \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --patch-tokenizer-type LLamaTokenizer \
        --swiglu \
        --normalization RMSNorm \
        --use-llama2-rotary-position-embeddings \
        --position-embedding-type rope \
        --untie-embeddings-and-output-weights \
        --rotary-base 100000 \
        --rotary-scale-factor 4 \
        --disable-bias-linear
        "

run_cmd="torchrun $DISTRIBUTED_ARGS ../llama2/evaluate_megatron_llama.py
 ${megatron_options} ${pr_options} ${load_options} ${te_options} ${activation_checkpoint_options} ${do_options} ${flash_options} ${sp_options} ${gqa_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
