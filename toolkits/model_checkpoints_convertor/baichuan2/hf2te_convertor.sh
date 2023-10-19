#!/bin/bash

set -e
START_TIME=$SECONDS

MEGATRON_PATH=$1
SOURCE_CKPT_PATH=$2
TARGET_CKPT_PATH=$3
TP=$4
PP=$5
MN=$6 #baichuan2-7b
EXTRA_VOCAB_SIZE=$7

export PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH

python hf2te.py \
--load_path ${SOURCE_CKPT_PATH} \
--save_path ${TARGET_CKPT_PATH} \
--target_params_dtype fp16 \
--megatron-path ${MEGATRON_PATH} \
--target_tensor_model_parallel_size ${TP} \
--target_pipeline_model_parallel_size ${PP} \
--model_name ${MN} \
--extra_num_vocabs ${EXTRA_VOCAB_SIZE} \

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
