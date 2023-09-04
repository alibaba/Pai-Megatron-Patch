#!/bin/bash
# bash run_text_generation_megatron_falcon40b.sh dsw /workspace/Megatron-LM /workspace/PAI-Megatron-Patch /mnt/llama-ckpts/Ziya-LLaMA-13B-to-megatron-tp1-pp1 13B 1 1 1024 80 16 fp16 0 512 512 /mnt/llama-datasets/cn_input.txt /mnt/llama-datasets/cn_output.txt 0.85 1 1
set -e
ENV=$1
export CUDA_VISIBLE_DEVICES=0
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
MEGATRON_PATH=$2
MEGATRON_PATCH_PATH=$3
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:$PYTHONPATH

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

CHECKPOINT_PATH=$4
MODEL_SIZE=$5
TP=$6
BS=$7
SEQ_LEN=$8
PAD_LEN=$9
EXTRA_VOCAB_SIZE=${10}
PR=${11}
TOP_K=${12}
INPUT_SEQ_LEN=${13}
OUTPUT_SEQ_LEN=${14}
INPUT_FILE=${15}
OUTPUT_FILE=${16}
TOP_P=${17}
TEMPERATURE=${18}
# set this penalty between 1.1 and 1.5 to reduce repetition, default is 1.2
REPETITION_PENALTY=${19}

if [ $MODEL_SIZE = 7B ]; then

NUM_LAYERS=32
HIDDEN_SIZE=4544
NUM_ATTN_HEADS=71

fi

if [ $CHECKPOINT_PATH != none ]; then
    load_options=" \
		    --load $CHECKPOINT_PATH"
fi

if [ $INPUT_FILE = none ]; then
    input_options=" \
		               "
else
    input_options=" \
        --text-generate-output-file ${OUTPUT_FILE}\
        --text-generate-input-file ${INPUT_FILE} \
        "
fi

if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16"
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
fi

rapidformer_options="  \
        --micro-batch-size ${BS} \
        --num-layers ${NUM_LAYERS}  \
        --hidden-size ${HIDDEN_SIZE}  \
        --num-attention-heads ${NUM_ATTN_HEADS}  \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size 1 \
        --no-load-optim \
        --no-load-rng \
        --DDP-impl local\
        --top-p ${TOP_P} \
        --temperature ${TEMPERATURE}  \
        --top-k ${TOP_K} \
        --input-len ${INPUT_SEQ_LEN} \
        --out-seq-length ${OUTPUT_SEQ_LEN}  \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --max-padding-length ${PAD_LEN} \
        --use-distributed-optimizer \
        --position-embedding-type rotary \
        --patch-tokenizer-type FalconTokenizer \
        --attention-head-type multiquery \
        --disable-bias-linear \
        --repetition-penalty ${REPETITION_PENALTY} \
    "

run_cmd="torchrun $DISTRIBUTED_ARGS generate_text_megatron_falcon.py
 ${rapidformer_options} ${load_options} ${input_options} ${pr_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
