#!/bin/bash
# transformers2megatron
# bash model_convertor_huggingface_megatron.sh ../../Megatron-LM/ ../../bloomwcp-shrink ../../bloomwcp-shrink_mg_test 1 1 false
# megatron2transformers
# bash model_convertor_huggingface_megatron.sh ../../Megatron-LM/ ../../bloomwcp-shrink_mg_test/release/ ../../bloomwcp-shrink_mg2hf-wotrain 1 1 true
set -e
START_TIME=$SECONDS

MEGATRON_PATH=$1
export PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH

SOURCE_CKPT_PATH=$2
TARGET_CKPT_PATH=$3
TP=$4
PP=$5
mg2hf=$6

if [ $mg2hf = true ]; then
    do_options="
                --convert_checkpoint_from_megatron_to_transformers
    "

elif [ $mg2hf = false ]; then
    do_options=""
fi

python checkpoint_reshaping_and_interoperability.py \
--load_path ${SOURCE_CKPT_PATH} \
--save_path ${TARGET_CKPT_PATH} \
--target_params_dtype fp16 \
--megatron-path ${MEGATRON_PATH} \
--target_tensor_model_parallel_size ${TP} \
--target_pipeline_model_parallel_size ${PP}\
${do_options}

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
