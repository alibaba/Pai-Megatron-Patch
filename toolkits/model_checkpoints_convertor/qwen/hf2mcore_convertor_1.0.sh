#!/bin/bash
# bash hf2mcore_convertor_1.0.sh ../../.. /mnt/qwen-ckpts/Qwen-1_8B /mnt/qwen-ckpts/Qwen-1_8B-to-mcore-tp1-pp1/ 1 1 8 8 qwen-1.8b false
set -e
START_TIME=$SECONDS
export CUDA_VISIBLE_DEVICES=7
MEGATRON_PATCH_PATH=$1
MEGATRON_PATH=${MEGATRON_PATCH_PATH}/Megatron-LM-240126
SOURCE_CKPT_PATH=$2
TARGET_CKPT_PATH=$3
TP=$4
PP=$5
EP=$6
WS=$7
MN=$8 #qwen-1.8b
MG2HF=$9

if [ $MG2HF = true ]; then
    convert_options="
                --convert_checkpoint_from_megatron_to_transformers
    "
elif [ $MG2HF = false ]; then
    convert_options=""
fi


export PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH

python hf2mcore_1.0.py \
--load_path ${SOURCE_CKPT_PATH} \
--save_path ${TARGET_CKPT_PATH} \
--target_params_dtype bf16 \
--target_tensor_model_parallel_size ${TP} \
--target_pipeline_model_parallel_size ${PP} \
--target_expert_model_parallel_size ${EP} \
--world_size ${WS} \
--model_name ${MN} \
${convert_options}

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
