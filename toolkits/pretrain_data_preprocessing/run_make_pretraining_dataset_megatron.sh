#! /bin/bash
START_TIME=$SECONDS

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATH}:${MEGATRON_PATH}/Megatron-LM-240405

input_data_dir=$1
tokenizer=$2
output_data_dir=$3
load_dir=$4

INPUT="${input_data_dir}"

if [ $tokenizer = "Qwen2Tokenizer" ]; then
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

elif [ $tokenizer = "DeepSeekV2Tokenizer" ]; then
  python preprocess_data_megatron.py \
  --input ${INPUT} \
  --output-prefix ${output_data_dir}/SlimPajama_deepseekv2bpe \
  --patch-tokenizer-type DeepSeekV2Tokenizer \
  --tokenizer-type GPT2BPETokenizer \
  --load ${load_dir} \
  --workers 8 \
  --partitions 1 \
  --keep-sequential-samples \
  --append-eod

elif [ $tokenizer = "LLamaTokenizer" ]; then
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
