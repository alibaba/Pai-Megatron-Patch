#!/bin/bash
# bash hf2mcore_moe_convertor.sh ../../../Megatron-LM-240126/ /mnt/qwen-ckpts/Qwen-1_8B /mnt/qwen-ckpts/Qwen-1_8B-to-mcore-tp4-ep4/ 4 1 4 16 qwen-1.8b false
set -e
START_TIME=$SECONDS

MEGATRON_PATH=$1
SOURCE_CKPT_PATH=$2
TARGET_CKPT_PATH=$3
TP=$4
PP=$5
EP=$6
WS=$7
MN=$8 #qwen-1.8b
MG2HF=$9

if [ $MG2HF = true ]; then
    do_options="
                --convert_checkpoint_from_megatron_to_transformers
    "
elif [ $MG2HF = false ]; then
    do_options=""
fi


export PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH

python hf2mcore_moe.py \
--load_path ${SOURCE_CKPT_PATH} \
--save_path ${TARGET_CKPT_PATH} \
--target_params_dtype fp16 \
--target_tensor_model_parallel_size ${TP} \
--target_pipeline_model_parallel_size ${PP} \
--target_expert_model_parallel_size ${EP} \
--world_size ${WS} \
--model_name ${MN} \
${do_options}

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
