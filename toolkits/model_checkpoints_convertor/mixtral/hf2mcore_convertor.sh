#!/bin/bash
# bash hf2mcore_convertor.sh ../../../Megatron-LM-231221 /mnt/workspace/latest/mixtral/Mixtral-8x7B-Instruct-v0.1-to-mcore-tp4-ep4/release/ /mnt/workspace/latest/mixtral/test 4 1 4 16 mixtral-8x7b true
set -e
START_TIME=$SECONDS

MEGATRON_PATH=$1
SOURCE_CKPT_PATH=$2
TARGET_CKPT_PATH=$3
TP=$4
PP=$5
EP=$6
WS=$7
MN=$8 #mixtral-8x7b or mistral-7b
MG2HF=$9

if [ $MG2HF = true ]; then
    do_options="
                --convert_checkpoint_from_megatron_to_transformers
    "
elif [ $MG2HF = false ]; then
    do_options=""
fi


export PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH

python hf2mcore.py \
--load_path ${SOURCE_CKPT_PATH} \
--save_path ${TARGET_CKPT_PATH} \
--target_params_dtype bf16 \
--target_tensor_model_parallel_size ${TP} \
--target_pipeline_model_parallel_size ${PP} \
--target_expert_model_parallel_size ${EP} \
--world_size ${WS} \
--model_name ${MN} \
${do_options}

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
