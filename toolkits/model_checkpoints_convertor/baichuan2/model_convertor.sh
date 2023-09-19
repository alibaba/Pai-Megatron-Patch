#!/bin/bash
# transformers to megatron
# sh model_convertor.sh /root/Megatron-LM-23.04/ /mnt/baichuan-ckpts/baichuan-13b-base/ /mnt/baichuan-ckpts/baichuan-13b-base-hf-to-megatron-tp1-pp1 1 1 baichuan-13b 0 false
# megatron to transformers
# sh model_convertor.sh ../../../../Megatron-LM/ ../../../../baichuan/baichuan-13b-base-hf-to-megatron-tp4-pp1/release/ ../../../../baichuan/baichuan2-13b-mg2hf41  4 1 baichuan2-13b 0 true
set -e
START_TIME=$SECONDS

MEGATRON_PATH=$1
SOURCE_CKPT_PATH=$2
TARGET_CKPT_PATH=$3
TP=$4
PP=$5
MN=$6 #baichuan2-7b, baichuan2-13b
EXTRA_VOCAB_SIZE=$7
mg2hf=$8

if [ $mg2hf = true ]; then
    do_options="
                --convert_checkpoint_from_megatron_to_transformers
    "
elif [ $mg2hf = false ]; then
    do_options=""
fi

export PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH

python checkpoint_reshaping_and_interoperability.py \
--load_path ${SOURCE_CKPT_PATH} \
--save_path ${TARGET_CKPT_PATH} \
--target_params_dtype fp16 \
--megatron-path ${MEGATRON_PATH} \
--target_tensor_model_parallel_size ${TP} \
--target_pipeline_model_parallel_size ${PP} \
--model_name ${MN} \
--extra_num_vocabs ${EXTRA_VOCAB_SIZE} \
${do_options}

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
