#! /bin/bash
set -ex
# 请在此处设置原始数据所在路径
data_dir=/mnt/mixtral-datasets/wudao_200g

#开始数据清洗流程
dataset_dir=$(dirname $data_dir)
mkdir -p ${dataset_dir}/cleaned_wudao_dataset
cd ${dataset_dir}/cleaned_wudao_dataset
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama2-codes/preprocess_wudao2.py
# 此处与上一节不同，增加了key参数设为text
python preprocess_wudao2.py -i ${data_dir} -o ${dataset_dir}/cleaned_wudao_dataset -k text -p 32

# 合并清洗后的数据
mkdir -p ${dataset_dir}/wudao
cd ${dataset_dir}/wudao
find ${dataset_dir}/cleaned_wudao_dataset -name "*.json" -exec cat {} + > ${dataset_dir}/wudao/merged_wudao_cleaned.json
rm -rf ${dataset_dir}/cleaned_wudao_dataset

# 此处设置分块数为10，如数据处理慢可设置稍大
NUM_PIECE=10
# 对merged_wudao_cleaned.json文件进行处理
mkdir -p ${dataset_dir}/cleaned_zst/
# 查询数据总长度，对数据进行拆分
NUM=$(sed -n '$=' ${dataset_dir}/wudao/merged_wudao_cleaned.json)
echo "total line of dataset is $NUM, data will be split into $NUM_PIECE pieces for processing"
NUM=`expr $NUM / $NUM_PIECE`
echo "each group is processing $NUM sample"
split_dir=${dataset_dir}/split
mkdir $split_dir
split -l $NUM --numeric-suffixes --additional-suffix=.jsonl ${dataset_dir}/wudao/merged_wudao_cleaned.json $split_dir/

# 数据压缩
o_path=${dataset_dir}/cleaned_zst/
mkdir -p $o_path
files=$(ls $split_dir/*.jsonl)
for filename in $files
do
   f=$(basename $filename)
   zstd -z $filename -o $o_path/$f.zst &
done
rm -rf $split_dir
rm ${dataset_dir}/wudao/merged_wudao_cleaned.json