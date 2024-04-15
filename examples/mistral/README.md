# Table of Contents
   * [安装](#安装)
   * [数据集&模型下载](#数据集&模型下载)
   * [Megatron-LM-Dense模型训练流程](#Megatron-LM-Dense模型训练流程)
      * [模型格式转换](#Megatron-LM-Dense模型格式转换)
      * [继续预训练](#Megatron-LM-Dense继续预训练)
      * [指令微调](#Megatron-LM-Dense指令微调)
   * [下游任务评估](#下游任务评估)
      * [Megatron-LM模型格式转换](#Megatron-LM-Dense模型转成Huggingface格式)
      * [Megatron-Core模型格式转换](#Megatron-Core-Dense模型转成Huggingface格式)
      * [运行评估工具](#运行评估工具)

# 安装
推荐使用英伟达提供的官方镜像 nvcr.io/nvidia/pytorch:23.12-py3 来创建容器

```bash
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
cd Pai-Megatron-Patch
pip install -e .
```

# 数据集&模型下载
```bash
cd /mnt
mkdir mistral-ckpts
cd mistral-ckpts
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-ckpts/Mistral-7B-v0.1.tgz
tar -zxf Mistral-7B-v0.1.tgz

wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/wudao_mistralbpe_content_document_small.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/wudao_mistralbpe_content_document_small.idx

wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/alpaca_zh-mistral-train.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/alpaca_zh-mistral-valid.json


```

# Megatron-LM-Dense模型训练流程

## Megatron-LM-Dense模型格式转换
使用我们提供的模型转换脚本，将huggingface格式的模型文件转换为megatron格式：
```bash


cd /workspace/PAI-Megatron-Patch/toolkits/model_checkpoints_convertor/mistral
sh model_convertor.sh \
../../../     \
/mnt/mistral-ckpts/Mistral-7B-v0.1    \
/mnt/mistral-ckpts/Mistral-7B-v0.1-hf-to-megatron-tp4-pp1  \
4  \
1  \
mistral-7b \
0 \
false
```

## Megatron-LM-Dense继续预训练
运行run_pretrain_megatron_mistral.sh脚本，需要传入的参数列表如下
```
ENV=$1                          # 运行环境: dlc, dsw
MEGATRON_PATCH_PATH=$2          # 设置Megatron Patch的代码路径
MODEL_SIZE=$3                   # 模型结构参数量级：7B, 13B
BATCH_SIZE=$4                   # 每卡训练一次迭代样本数: 4, 8
GLOBAL_BATCH_SIZE=$5            # 全局batch size
LR=$6                           # 学习率: 1e-5, 5e-5
MIN_LR=$7                       # 最小学习率: 1e-6, 5e-6
SEQ_LEN=$8                      # 序列长度
PAD_LEN=$9                      # Padding长度：100
EXTRA_VOCAB_SIZE=${10}          # 词表扩充大小
PR=${11}                        # 训练精度: fp16, bf16
TP=${12}                        # 模型并行度
PP=${13}                        # 流水并行度
AC=${14}                        # 激活检查点模式: sel, full
DO=${15}                        # 是否使用Megatron版Zero-1降显存优化器: true, false
FL=${16}                        # 是否使用Flash Attention: true, false
SP=${17}                        # 是否使用序列并行: true, false
TE=${18}                        # 是否使用Transformer Engine: true, false
SAVE_INTERVAL=${19}             # 保存ckpt的间隔
DATASET_PATH=${20}              # 训练数据集路径
PRETRAIN_CHECKPOINT_PATH=${21}  # 预训练模型路径
TRAIN_TOKENS=${22}              # 训练token数
WARMUP_TOKENS=${23}             # 预热token数
OUTPUT_BASEPATH=${24}           # 训练输出文件路径
```
单机运行示例如下：
```bash
cd /workspace/PAI-Megatron-Patch/examples/mistral
sh run_pretrain_megatron_mistral.sh  \
dsw  \
../../ \
7B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
0   \
bf16  \
4   \
1  \
sel  \
true   \
false  \
false   \
false   \
100000  \
wudao_mistralbpe_content_document_small   \
/mnt/mistral-ckpts/Mistral-7B-v0.1-hf-to-megatron-tp4-pp1  \
100000000   \
10000   \
/mnt/output_megatron_mistral
```

## Megatron-LM-Dense指令微调
运行run_finetune_megatron_mistral.sh脚本，需要传入的参数列表如下
```
ENV=$1                          # 运行环境: dlc, dsw
MEGATRON_PATCH_PATH=$2          # 设置Megatron Patch的代码路径
MODEL_SIZE=$3                   # 模型结构参数量级: 7B, 13B
BATCH_SIZE=$4                   # 每卡训练一次迭代样本数: 4, 8
LR=$5                           # 学习率: 1e-5, 5e-5
MIN_LR=$6                       # 最小学习率: 1e-6, 5e-6
SEQ_LEN=$7                      # 序列长度
PAD_LEN=$8                      # Padding长度：100
EXTRA_VOCAB_SIZE=$9          # 词表扩充大小
PR=${10}                        # 训练精度: fp16, bf16
TP=${11}                        # 模型并行度
PP=${12}                        # 流水并行度
AC=${13}                        # 激活检查点模式: sel, full
DO=${14}                        # 是否使用Megatron版Zero-1降显存优化器: true, false
FL=${15}                        # 是否使用Flash Attention: true, false
SP=${16}                        # 是否使用序列并行: true, false
TE=${17}                        # 是否使用Transformer engine: true, false
TRAIN_DATASET_PATH=${18}        # 训练数据集路径
VALID_DATASET_PATH=${19}        # 验证数据集路径
PRETRAIN_CHECKPOINT_PATH=${20}  # 预训练模型路径
EPOCH=${21}                     # 训练迭代轮次
OUTPUT_BASEPATH=${22}           # 训练输出文件路径
```
DSW单机运行示例如下：
```bash
cd /workspace/PAI-Megatron-Patch/examples/mistral
sh run_finetune_megatron_mistral.sh  \
dsw  \
../../ \
7B     \
1      \
1e-5   \
1e-6   \
128   \
128     \
0      \
bf16   \
4      \
1      \
sel    \
true   \
false  \
false  \
false \
alpaca_zh-mistral-train.json   \
alpaca_zh-mistral-valid.json   \
/mnt/mistral-ckpts/Mistral-7B-v0.1-hf-to-megatron-tp4-pp1   \
2   \
/mnt/output_megatron_mistral/
```

# 下游任务评估

## Megatron-LM-Dense模型转成Huggingface格式

## Megatron-Core-Dense模型转成Huggingface格式

## 运行评估工具