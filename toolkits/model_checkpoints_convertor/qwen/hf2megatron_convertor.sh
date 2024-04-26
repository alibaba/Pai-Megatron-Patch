#!/bin/bash
# bash model_convertor.sh ../../../../Megatron-LM/ ../../../../qwen-14b-hf-to-mg-tp2-pp1/release/ ../../../../qwen-14b-mg2hf21 2 1 qwen-14b 0 true
# bash model_convertor.sh ../../../Megatron-LM-231007/ ../../../../qianwen/models--Qwen--Qwen1.5-7B-Chat/ ../../../../qianwen/models--Qwen--Qwen1.5-7B-Chat-hf2mg41/ 4 1 qwen1.5 0 false
# bash model_convertor.sh ../../../Megatron-LM-231007/ ../../../../qianwen/models--Qwen--Qwen1.5-7B-Chat-hf2mg41/release/ ../../../../qianwen/models--Qwen--Qwen1.5-7B-Chat-mg2hf41/ 4 1 qwen1.5 0 true
set -e
START_TIME=$SECONDS

MEGATRON_PATH=$1
SOURCE_CKPT_PATH=$2
TARGET_CKPT_PATH=$3
TP=$4
PP=$5
MN=$6 #qwen-7b,qwen-14b,qwen-72b;qwen1.5-0.5b,qwen1.5-1.8b,qwen1.5-4b,qwen1.5-7b,qwen1.5-14b,qwen1.5-72b
EXTRA_VOCAB_SIZE=$7 # 0 for all models
mg2hf=$8

if [ $mg2hf = true ]; then
    do_options="
                --convert_checkpoint_from_megatron_to_transformers
    "
elif [ $mg2hf = false ]; then
    do_options=""
fi

export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATH}:${MEGATRON_PATH}/Megatron-LM-231007

if [[ "$MN" == *"qwen1.5"* ]]; then

python hf2megatron_qwen1.5.py \
--load_path ${SOURCE_CKPT_PATH} \
--save_path ${TARGET_CKPT_PATH} \
--target_params_dtype bf16 \
--megatron-path ${MEGATRON_PATH} \
--target_tensor_model_parallel_size ${TP} \
--target_pipeline_model_parallel_size ${PP} \
--model_name ${MN} \
--extra_num_vocabs ${EXTRA_VOCAB_SIZE} \
${do_options}

else
    
python hf2megatron_qwen1.0.py \
--load_path ${SOURCE_CKPT_PATH} \
--save_path ${TARGET_CKPT_PATH} \
--target_params_dtype fp16 \
--megatron-path ${MEGATRON_PATH} \
--target_tensor_model_parallel_size ${TP} \
--target_pipeline_model_parallel_size ${PP} \
--model_name ${MN} \
--extra_num_vocabs ${EXTRA_VOCAB_SIZE} \
${do_options}

fi

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
