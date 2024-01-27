#! /bin/bash
START_TIME=$SECONDS
MEGATRON_PATH=$1
MEGATRON_PATCH_PATH=$2
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:$PYTHONPATH
input_data_dir=$3
tokenizer=$4
output_data_dir=$5
load_dir=$6

INPUT="${input_data_dir}"



if [ $tokenizer = "jiebabpe" ]; then

if [ ! -f tokenizer.json ]; then
  wget https://easynlp-dev.oss-cn-zhangjiakou.aliyuncs.com/225247/RapidformerPro/tokenizer.json
fi

python preprocess_data.py \
  --input ${INPUT} \
  --language zh \
  --output-prefix ${output_data_dir}/wudao_jiebabpe \
  --dataset-impl mmap \
  --vocab tokenizer.json \
  --patch-tokenizer-type JiebaBPETokenizer \
  --load ${load_dir} \
  --workers 16 \
  --append-eod

elif [ $tokenizer = "bloombpe" ]; then

  python preprocess_data.py \
  --input ${INPUT} \
  --language zh \
  --output-prefix ${output_data_dir}/wudao_bloombpe \
  --dataset-impl mmap \
  --patch-tokenizer-type BloomTokenizerFromHF \
  --load ${load_dir} \
  --workers 16 \
  --append-eod

elif [ $tokenizer = "glmchinesebpe" ]; then

  python preprocess_data.py \
  --input ${INPUT} \
  --language zh \
  --output-prefix ${output_data_dir}/wudao_glmchinesebpe \
  --dataset-impl mmap \
  --patch-tokenizer-type GLM10BZHTokenizerFromHF \
  --load ${load_dir} \
  --workers 16 \
  --append-eod

elif [ $tokenizer = "glm130bbpe" ]; then

  python preprocess_data.py \
  --input ${INPUT} \
  --language zh \
  --output-prefix ${output_data_dir}/wudao_glm130bbpe \
  --dataset-impl mmap \
  --patch-tokenizer-type IcetkGLM130BTokenizer \
  --load ${load_dir} \
  --workers 16 \
  --append-eod

elif [ $tokenizer = "llamabpe" ]; then

  python preprocess_data.py \
  --input ${INPUT} \
  --language zh \
  --output-prefix ${output_data_dir}/wudao_llamabpe \
  --dataset-impl mmap \
  --patch-tokenizer-type LLamaTokenizer \
  --load ${load_dir} \
  --workers 16 \
  --append-eod

elif [ $tokenizer = "falconbpe" ]; then

  python preprocess_data.py \
  --input ${INPUT} \
  --language zh \
  --output-prefix ${output_data_dir}/wudao_falconbpe \
  --dataset-impl mmap \
  --patch-tokenizer-type FalconTokenizer \
  --load ${load_dir} \
  --workers 16 \
  --append-eod

elif [ $tokenizer = "galacticabpe" ]; then

  python preprocess_data.py \
  --input ${INPUT} \
  --language zh \
  --output-prefix ${output_data_dir}/wudao_galacticabpe \
  --dataset-impl mmap \
  --patch-tokenizer-type OPTTokenizer \
  --load ${load_dir} \
  --workers 16 \
  --append-eod

elif [ $tokenizer = "starcoderbpe" ]; then
  python preprocess_data.py \
  --input ${INPUT} \
  --language zh \
  --output-prefix ${output_data_dir}/wudao_starcoderbpe \
  --dataset-impl mmap \
  --patch-tokenizer-type StarcoderTokenizerFromHF \
  --load ${load_dir} \
  --workers 16 \
  --append-eod

elif [ $tokenizer = "qwenbpe" ]; then
  python preprocess_data.py \
  --input ${INPUT} \
  --language zh \
  --output-prefix ${output_data_dir}/wudao_qwenbpe \
  --dataset-impl mmap \
  --patch-tokenizer-type QwenTokenizer \
  --load ${load_dir} \
  --workers 16 \
  --append-eod

elif [ $tokenizer = "mistralbpe" ]; then
  python preprocess_data.py \
  --input ${INPUT} \
  --language zh \
  --output-prefix ${output_data_dir}/wudao_mistralbpe_small \
  --dataset-impl mmap \
  --patch-tokenizer-type MistralTokenizer \
  --load ${load_dir} \
  --workers 16 \
  --append-eod

fi

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
