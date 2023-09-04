#!/bin/bash
# bash run_text_generation_megatron_starcoder.sh dsw /workspace/Megatron-LM /workspace/PAI-Megatron-Patch /mnt/starcoder 16B 1 1 1024 80 fp16 0 512 512 /mnt/datasets/cn_input.txt /mnt/datasets/cn_output.txt 0.85 1 1
set -e
ENV=$1
export FORCE_CUDA="1"
export CUDA_VISIBLE_DEVICES=0,1,2,3
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
GPUS_PER_NODE=4
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
PR=${10}
TOP_K=${11}
INPUT_SEQ_LEN=${12}
OUTPUT_SEQ_LEN=${13}
INPUT_FILE=${14}
OUTPUT_FILE=${15}
TOP_P=${16}
TEMPERATURE=${17}
# set this penalty between 1.1 and 1.5 to reduce repetition, default is 1.2
REPETITION_PENALTY=${18}

if [ $MODEL_SIZE = 16B ]; then

NUM_LAYERS=40
HIDDEN_SIZE=6144
NUM_ATTN_HEADS=48
INTERMEDIATE_SIZE=24576

elif [ $MODEL_SIZE = 7B ]; then

NUM_LAYERS=42
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
INTERMEDIATE_SIZE=16384

elif [ $MODEL_SIZE = 3B ]; then

NUM_LAYERS=36
HIDDEN_SIZE=2816
NUM_ATTN_HEADS=22
INTERMEDIATE_SIZE=11264

elif [ $MODEL_SIZE = 1B ]; then

NUM_LAYERS=24
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
INTERMEDIATE_SIZE=8192

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
        --intermediate-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings 8192 \
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
        --max-padding-length ${PAD_LEN} \
        --use-distributed-optimizer \
        --attention-head-type multiquery \
        --patch-tokenizer-type StarcoderTokenizerFromHF \
        --repetition-penalty ${REPETITION_PENALTY} \
    "

run_cmd="torchrun $DISTRIBUTED_ARGS generate_text_megatron_starcoder.py
 ${rapidformer_options} ${load_options} ${input_options} ${pr_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
