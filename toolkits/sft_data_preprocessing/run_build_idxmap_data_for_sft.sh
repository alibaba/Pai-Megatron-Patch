#! /bin/bash
START_TIME=$SECONDS

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATH}:${MEGATRON_PATH}/Megatron-LM-240612

input_data_path=$1
tokenizer=$2
seq_len=$3
output_data_path=$4
load_dir=$5

if [ $tokenizer = "Qwen2Tokenizer" ]; then
  python build_idxmap_data.py\
  --input ${input_data_path} \
  --output-prefix ${output_data_path} \
  --patch-tokenizer-type Qwen2Tokenizer \
  --load ${load_dir} \
  --seq-length ${seq_len} \
  --workers 8 \
  --partitions 1 \

elif [ $tokenizer = "LLama3Tokenizer" ]; then
  python build_idxmap_data.py\
  --input ${input_data_path} \
  --output-prefix ${output_data_path} \
  --patch-tokenizer-type LLama3Tokenizer \
  --load ${load_dir} \
  --seq-length ${seq_len} \
  --workers 8 \
  --partitions 1 \

fi

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
