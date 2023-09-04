#!/bin/bash
# bash run_text_generation_megatron_bloom.sh dsw /workspace/Megatron-LM /workspace/PAI-Megatron-Patch bloombpe-c /mnt/bloom-ckpts/bloomwcp-shrink_mg_test 7.1B 1 1 1024 fp16 10 512 512 /mnt/bloom-datasets/pred_input.jsonl /mnt/bloom-datasets/bloom_pred.txt 0 1.0 1.2
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

TOKENIZER=$4
CHECKPOINT_PATH=$5
MODEL_SIZE=$6
TP=$7
BS=$8
SEQ_LEN=$9
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

if [ $TOKENIZER = jiebabpe ]; then

if [ ! -f tokenizer.json ]; then
  wget https://easynlp-dev.oss-cn-zhangjiakou.aliyuncs.com/225247/RapidformerPro/tokenizer.json
fi

    tokenizer_options=" \
        --tokenizer-type JiebaBPETokenizer \
        --vocab-file tokenizer.json
        "

elif [ $TOKENIZER = bloombpe-c ]; then
    tokenizer_options=" \
		    --patch-tokenizer-type BloomTokenizerFromCustom
		    "

elif [ $TOKENIZER = bloombpe-h ]; then
    tokenizer_options=" \
		    --patch-tokenizer-type BloomTokenizerFromHF
		    "
fi

if [ $MODEL_SIZE = 1.1B ]; then

NUM_LAYERS=24
HIDDEN_SIZE=1536
NUM_ATTN_HEADS=16

elif [ $MODEL_SIZE = 1.7B ]; then

NUM_LAYERS=24
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16

elif [ $MODEL_SIZE = 7.1B ]; then

NUM_LAYERS=30
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32

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
       --use-distributed-optimizer \
       --seq-length ${SEQ_LEN} \
       --max-position-embeddings ${SEQ_LEN} \
       --no-load-optim \
       --DDP-impl local\
       --top-p ${TOP_P} \
       --temperature ${TEMPERATURE}  \
       --top-k ${TOP_K} \
       --input-len ${INPUT_SEQ_LEN} \
       --out-seq-length ${OUTPUT_SEQ_LEN}  \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size 1 \
       --embed-layernorm \
       --position-embedding-type alibi \
       --repetition-penalty ${REPETITION_PENALTY} \
    "

run_cmd="torchrun $DISTRIBUTED_ARGS generate_text_bloom.py ${tokenizer_options}
 ${rapidformer_options} ${load_options} ${input_options} ${pr_options} "


echo ${run_cmd}
eval ${run_cmd}
set +x
