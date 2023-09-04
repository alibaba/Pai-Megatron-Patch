#!/bin/bash
set -e
START_TIME=$SECONDS

MEGATRON_PATH=$1
export PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH

HF_CKPT_PATH=$2
MG_CKPT_PATH=$3
TP=$4
PP=$5

python checkpoint_reshaping_and_interoperability.py \
--load_path ${HF_CKPT_PATH} \
--save_path ${MG_CKPT_PATH} \
--target_params_dtype fp32 \
--megatron-path ${MEGATRON_PATH} \
--target_tensor_model_parallel_size ${TP} \
--target_pipeline_model_parallel_size ${PP} \

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
