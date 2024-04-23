## 数据预处理
建议在PAI灵骏智算服务中的DSW实例中准备预训练数据，以下以中文wudao2.0数据集的准备流程为例，给出数据预处理指引：
下载WuDaoCorpora2.0开源数据集到/mnt/workspace/llama3-datasets工作目录下，我们提供了部分样例数据作为示例，用户可通过以下命令下载和解压：
```shell
wget https://atp-modelzoo.oss-cn-hangzhou.aliyuncs.com/release/datasets/WuDaoCorpus2.0_base_sample.tgz
tar zxvf WuDaoCorpus2.0_base_sample.tgz 
```
假设解压后的文件夹命名为wudao_200g，该文件夹中的**原始**wudao数据集的格式和大小如下截图所示：
![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2023/png/226643/1681404415062-92c59f2f-380a-4357-baf8-6f626ad5a217.png#clientId=u90be3297-6831-4&from=paste&height=213&id=NOUVT&originHeight=426&originWidth=1054&originalType=binary&ratio=2&rotation=0&showTitle=false&size=154924&status=done&style=none&taskId=ua99b6661-8759-4a1e-8b97-2cd86736261&title=&width=527)
我们为Megatron-LM训练准备了数据预处理流程，您可以根据自己的需要选择不同的处理方式。
#### Megatron-LM训练数据准备
mmap数据是一种预先执行tokenize处理的数据格式，可以极大减少训练微调过程中等待数据读入的时间，当数据量极大时，优势显著。

1. 对Wudao数据执行数据集清洗并进行文件格式转换，具体流程可参考如下的bash脚本，最终生成汇总的**merged_wudao_cleaned.json**。
```bash
#! /bin/bash
set -ex
# 请在此处设置原始数据所在路径
data_dir=/mnt/workspace/llama3-datasets/wudao_200g

#开始数据清洗流程
dataset_dir=$(dirname $data_dir)
mkdir -p ${dataset_dir}/cleaned_wudao_dataset
cd ${dataset_dir}/cleaned_wudao_dataset
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama2-codes/preprocess_wudao2.py
# 此处与上一节不同，增加了key参数设为text
python preprocess_wudao2.py -i ${data_dir} -o ${dataset_dir}/cleaned_wudao_dataset -k text -p 32

# 合并清洗后的数据
mkdir ${dataset_dir}/wudao
cd ${dataset_dir}/wudao
find ${dataset_dir}/cleaned_wudao_dataset -name "*.json" -exec cat {} + > ${dataset_dir}/wudao/merged_wudao_cleaned.json
rm -rf ${dataset_dir}/cleaned_wudao_dataset

```
脚本执行完成后，llama3-datasets内部文件结构如下，新增一个wudao文件夹：
```bash
llama3-datasets
├── wudao_200g 
└── wudao
    └── merged_wudao_cleaned.json
```

2. 利用第一节生成的**merged_wudao_cleaned.json**文件，将数据拆分成若干组并压缩，便于后续实现多线程处理：
```bash
apt-get update
apt-get install zstd

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

```
脚本执行完成后，llama3-datasets内部文件结构如下，新增一个cleaned_zst文件夹，每个子文件夹里有10个压缩文件：
```bash
llama3-datasets
├── wudao_200g
├── wudao
└── cleaned_zst
    ├── 00.jsonl.zst
		│   ...
    └── 09.jsonl.zst
```

3. 制作MMAP格式预训练数据集。

前往[Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch)开源网站获取Megatron模型训练工具Pai-Megatron-Patch源代码并拷贝到工作目录/mnt/workspace/下。
```bash
# 开源网站获取训练代码
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
```

在DSW的Terminal中进入代码目录：/mnt/workspace/Pai-Megatron-Patch/toolkits/pretrain_data_preprocessing。查看run_make_pretraining_dataset.sh脚本内容。里面有5个启动参数需要在运行时输入，具体参数列表如下：
```
MEGATRON_PATCH_PATH=$1             # 设置Megatron Patch的代码路径
input_data_dir=$2                  # 打包后的wudao数据集的文件夹路径
tokenizer=$3                       # llamabpe
output_data_dir=$4                 # 输出到bin和idx文件目录  
load_dir=$5                        # tokenizer_config.json文件路径
```
运行示例如下所示：
```bash

# 请在此处设置数据集路径和工作路径
export dataset_dir=/mnt/workspace/llama3-datasets
export WORK_DIR=/mnt/workspace

# 分别为训练集、验证集生成mmap格式预训练数据集
cd ${WORK_DIR}/Pai-Megatron-Patch/toolkits/pretrain_data_preprocessing
bash run_make_pretraining_dataset.sh \
../.. \
${dataset_dir}/cleaned_zst/ \
llamabpe \
${dataset_dir}/ \
${WORK_DIR}/llama3-ckpts/Meta-Llama-3-8B
rm -rf ${dataset_dir}/cleaned_zst
```
脚本执行完成后，llama3-datasets内部文件结构如下，wudao文件夹里有2个名字相同后缀不同的mmap文件：
```bash
llama3-datasets
├── wudao_200g
└── wudao
   ├── wudao_llama3bpe_content_document.bin
   └── wudao_llama3bpe_content_document.idx
```
#### 小规模预处理数据下载试用
为方便用户试用，我们也提供了已经处理好的小规模数据，可直接下载使用
```bash
cd /mnt/workspace/llama3-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/wudao_llama3bpe_content_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/wudao_llama3bpe_content_document.idx
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/alpaca_zh-llama3-train.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/alpaca_zh-llama3-valid.json
```