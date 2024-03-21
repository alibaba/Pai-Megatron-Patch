#!/bin/bash
# bash hf2mcore_convertor_1.5_v1.sh ../../.. /mnt/qwen-ckpts/Qwen1.5-0.5B /mnt/qwen-ckpts/qwen1.5_0.5b_mcore_tp1_pp1_v1 1 1 qwen1.5-0.5B false
set -e
START_TIME=$SECONDS
export CUDA_VISIBLE_DEVICES=7
MEGATRON_PATCH_PATH=$1
MEGATRON_PATH=${MEGATRON_PATCH_PATH}/Megatron-LM-240126
SOURCE_CKPT_PATH=$2
TARGET_CKPT_PATH=$3
TP=$4
PP=$5
MN=$6 #qwen1.5-0.5B
MG2HF=$7

if [ $MG2HF = true ]; then
    convert_options="
                --convert_checkpoint_from_megatron_to_transformers
    "
elif [ $MG2HF = false ]; then
    convert_options=""
fi


export PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH

python hf2mcore_1.5_v1.py \
--load_path ${SOURCE_CKPT_PATH} \
--save_path ${TARGET_CKPT_PATH} \
--target_params_dtype bf16 \
--target_tensor_model_parallel_size ${TP} \
--target_pipeline_model_parallel_size ${PP} \
--model_name ${MN} \
${convert_options}

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
