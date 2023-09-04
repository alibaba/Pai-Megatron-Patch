#!/bin/bash
# huggingface to megatron
# bash model_convertor.sh /workspace/Megatron-LM/ /mnt/falcon-ckpts/falcon-7b-hf/ /mnt/falcon-ckpts/falcon-7b-hf-to-megatron-tp1-pp1 1 1 falcon-7b 0 false
# megatron to huggingface: you need to copy the corresponding tokenizer files into the save dir
# bash model_convertor.sh /workspace/Megatron-LM/ /mnt/falcon-ckpts/falcon-7b-hf-to-megatron-tp1-pp1/release/ /mnt/falcon-ckpts/falcon-7b-mg2hf/ 1 1 falcon-7b 0 true

set -e
START_TIME=$SECONDS

MEGATRON_PATH=$1
HF_CKPT_PATH=$2
MG_CKPT_PATH=$3
TP=$4
PP=$5
MN=$6 #falcon-7b, falcon-40b
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
--load_path ${HF_CKPT_PATH} \
--save_path ${MG_CKPT_PATH} \
--target_params_dtype fp16 \
--megatron-path ${MEGATRON_PATH} \
--target_tensor_model_parallel_size ${TP} \
--target_pipeline_model_parallel_size ${PP} \
--model_name ${MN} \
--extra_num_vocabs ${EXTRA_VOCAB_SIZE} \
${do_options}

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
