#!/bin/bash
set -e
START_TIME=$SECONDS

MEGATRON_PATH=$1
HF_CKPT_PATH=$2
MG_CKPT_PATH=$3
TP=$4
PP=$5
MN=$6 #galactica-6.7b, galactica-30b
EXTRA_VOCAB_SIZE=$7

export PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH

python checkpoint_reshaping_and_interoperability.py \
--load_path ${HF_CKPT_PATH} \
--save_path ${MG_CKPT_PATH} \
--target_params_dtype fp16 \
--megatron-path ${MEGATRON_PATH} \
--target_tensor_model_parallel_size ${TP} \
--target_pipeline_model_parallel_size ${PP} \
--model_name ${MN} \
--extra_num_vocabs ${EXTRA_VOCAB_SIZE}

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
