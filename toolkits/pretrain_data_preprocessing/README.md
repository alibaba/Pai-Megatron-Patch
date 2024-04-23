## Data Preprocessing
It is recommended to prepare pre-training data within the DSW instance of the PAI Lingjun smart computing service. For example, here are the steps for preparing the Chinese WuDao 2.0 dataset as a guide for data preprocessing:
Download the WuDaoCorpora2.0 open-source dataset to the /mnt/workspace/llama3-datasets working directory. We provide some sample data for reference, which users can download and extract using the following commands:
```shell
wget https://atp-modelzoo.oss-cn-hangzhou.aliyuncs.com/release/datasets/WuDaoCorpus2.0_base_sample.tgz
tar zxvf WuDaoCorpus2.0_base_sample.tgz 
```
Assuming the folder is named wudao_200g after extraction, the format and size of the **original** WuDao dataset in this folder are shown in the screenshot below:![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2023/png/226643/1681404415062-92c59f2f-380a-4357-baf8-6f626ad5a217.png#clientId=u90be3297-6831-4&from=paste&height=213&id=NOUVT&originHeight=426&originWidth=1054&originalType=binary&ratio=2&rotation=0&showTitle=false&size=154924&status=done&style=none&taskId=ua99b6661-8759-4a1e-8b97-2cd86736261&title=&width=527)
We have prepared a data preprocessing workflow for Megatron-LM training. You can choose different processing methods according to your needs.
#### Megatron-LM Training Data Preparation
mmap data is a data format that undergoes tokenization in advance, which can significantly reduce the waiting time for data input during the training and fine-tuning process. This advantage is particularly notable when dealing with very large volumes of data.
1. To clean the Wudao data and convert file formats, you can follow the process outlined in the bash script below. This script will ultimately generate a consolidated file:
**merged_wudao_cleaned.json**。
```bash
#! /bin/bash
set -ex
# 请在此处设置原始数据所在路径
data_dir=/mnt/workspace/llama3-datasets/wudao_200g

# Please set the path to the original data here
dataset_dir=$(dirname $data_dir)
mkdir -p ${dataset_dir}/cleaned_wudao_dataset
cd ${dataset_dir}/cleaned_wudao_dataset
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama2-codes/preprocess_wudao2.py
# Note: This section adds a 'key' parameter set to 'text', different from the previous section
python preprocess_wudao2.py -i ${data_dir} -o ${dataset_dir}/cleaned_wudao_dataset -k text -p 32

# Merge the cleaned data
mkdir ${dataset_dir}/wudao
cd ${dataset_dir}/wudao
find ${dataset_dir}/cleaned_wudao_dataset -name "*.json" -exec cat {} + > ${dataset_dir}/wudao/merged_wudao_cleaned.json
rm -rf ${dataset_dir}/cleaned_wudao_dataset

```
After the script is executed, the internal file structure of llama3-datasets is as follows, with a new folder named 'wudao' added:
```bash
llama3-datasets
├── wudao_200g 
└── wudao
    └── merged_wudao_cleaned.json
```

2. Using the **merged_wudao_cleaned.json** file generated in the first section, split the data into several groups and compress it to facilitate multithreaded processing in subsequent steps.：
```bash
apt-get update
apt-get install zstd

# Set the number of partitions to 10 here; if data processing is slow, consider setting a larger number.
NUM_PIECE=10

# Process the merged_wudao_cleaned.json file.
mkdir -p ${dataset_dir}/cleaned_zst/
# Calculate the total length of the data and split it.
NUM=$(sed -n '$=' ${dataset_dir}/wudao/merged_wudao_cleaned.json)
echo "total line of dataset is $NUM, data will be split into $NUM_PIECE pieces for processing"
NUM=`expr $NUM / $NUM_PIECE`
echo "each group is processing $NUM sample"
split_dir=${dataset_dir}/split
mkdir $split_dir
split -l $NUM --numeric-suffixes --additional-suffix=.jsonl ${dataset_dir}/wudao/merged_wudao_cleaned.json $split_dir/

# Data Compression
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

```
After the script is executed, the internal file structure of llama3-datasets is as follows, with a new folder named `cleaned_zst` added, containing 10 compressed files in each subfolder:
```bash
llama3-datasets
├── wudao_200g
├── wudao
└── cleaned_zst
    ├── 00.jsonl.zst
		│   ...
    └── 09.jsonl.zst
```

3. Create an MMAP format pre-training dataset.

Visit the [Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch) open-source website to download the Megatron model training tool Pai-Megatron-Patch source code and copy it to the working directory `/mnt/workspace/`.
```bash
# Obtain the training code from the open-source website
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
```

In the DSW Terminal, navigate to the code directory: `/mnt/workspace/Pai-Megatron-Patch/toolkits/pretrain_data_preprocessing`. Check the content of the `run_make_pretraining_dataset.sh` script. It requires five parameters to be entered at runtime, and the specific list of parameters is as follows:
```
MEGATRON_PATCH_PATH=$1             # 设置Megatron Patch的代码路径
input_data_dir=$2                  # 打包后的wudao数据集的文件夹路径
tokenizer=$3                       # llamabpe
output_data_dir=$4                 # 输出到bin和idx文件目录  
load_dir=$5                        # tokenizer_config.json文件路径
```
An example of running the script is as follows:
```bash

# Please set the dataset path and working directory here.
export dataset_dir=/mnt/workspace/llama3-datasets
export WORK_DIR=/mnt/workspace

# Generate MMAP format pre-training datasets for the training and validation sets respectively.
cd ${WORK_DIR}/Pai-Megatron-Patch/toolkits/pretrain_data_preprocessing
bash run_make_pretraining_dataset.sh \
../.. \
${dataset_dir}/cleaned_zst/ \
llamabpe \
${dataset_dir}/ \
${WORK_DIR}/llama3-ckpts/Meta-Llama-3-8B
rm -rf ${dataset_dir}/cleaned_zst
```
After the script is executed, the internal file structure of llama3-datasets is as follows, with the wudao folder containing two mmap files with the same name but different extensions:
```bash
llama3-datasets
├── wudao_200g
└── wudao
   ├── wudao_llama3bpe_content_document.bin
   └── wudao_llama3bpe_content_document.idx
```
#### Small-Scale Preprocessed Data Download for Trial Use
To facilitate user trials, we also provide preprocessed small-scale data that can be directly downloaded and used.
```bash
cd /mnt/workspace/llama3-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/wudao_llama3bpe_content_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/wudao_llama3bpe_content_document.idx
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/alpaca_zh-llama3-train.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/alpaca_zh-llama3-valid.json
```