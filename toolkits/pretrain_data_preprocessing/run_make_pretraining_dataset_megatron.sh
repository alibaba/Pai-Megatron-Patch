#! /bin/bash
START_TIME=$SECONDS
MEGATRON_PATCH_PATH=$1
MEGATRON_PATH=${MEGATRON_PATCH_PATH}/Megatron-LM-240405
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:$PYTHONPATH
input_data_dir=$2
tokenizer=$3
output_data_dir=$4
load_dir=$5

INPUT="${input_data_dir}"

if [ $tokenizer = "qwen2bpe" ]; then
  python preprocess_data_megatron.py \
  --input ${INPUT} \
  --output-prefix ${output_data_dir}/skypile_qwen2bpe \
  --patch-tokenizer-type Qwen2Tokenizer \
  --tokenizer-type GPT2BPETokenizer \
  --load ${load_dir} \
  --workers 2 \
  --partitions 2 \
  --keep-sequential-samples \
  --append-eod

elif [ $tokenizer = "llamabpe" ]; then
  python preprocess_data_megatron.py \
  --input ${INPUT} \
  --output-prefix ${output_data_dir}/SlimPajama_llamabpe \
  --patch-tokenizer-type LLamaTokenizer \
  --tokenizer-type GPT2BPETokenizer \
  --load ${load_dir} \
  --workers 16 \
  --partitions 1 \
  --keep-sequential-samples \
  --append-eod

fi

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
