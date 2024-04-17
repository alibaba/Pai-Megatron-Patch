# Table of Contents
   * [安装](#安装)
   * [数据集&模型下载](#数据集和模型下载)
   * [Megatron-LM-Dense模型训练流程](#Megatron-LM-Dense模型训练流程)
      * [模型格式转换](#Megatron-LM-Dense模型格式转换)
      * [继续预训练](#Megatron-LM-Dense继续预训练)
      * [指令微调](#Megatron-LM-Dense指令微调)
   * [Megatron-Core-Dense模型训练流程](#Megatron-Core-Dense模型训练流程)
      * [模型格式转换](#Megatron-Core-Dense模型格式转换)
      * [继续预训练](#Megatron-Core-Dense继续预训练)
      * [指令微调](#Megatron-Core-Dense指令微调)
   * [Megatron-Core-MoE模型训练流程](#Megatron-Core-MoE模型训练流程)
      * [模型格式转换](#Megatron-Core-MoE模型格式转换)
      * [继续预训练](#Megatron-Core-MoE继续预训练)
      * [指令微调](#Megatron-Core-MoE指令微调)
   * [MegaBlocks-MoE模型训练流程](#MegaBlocks-MoE模型训练流程)
      * [模型格式转换](#MegaBlocks-MoE模型格式转换)
      * [继续预训练](#MegaBlocks-MoE继续预训练)
      * [指令微调](#MegaBlocks-MoE指令微调)
   * [下游任务评估](#下游任务评估)
      * [Megatron-LM模型格式转换](#Megatron-LM-Dense模型转成Huggingface格式)
      * [Megatron-Core模型格式转换](#Megatron-Core-Dense模型转成Huggingface格式)
      * [运行评估工具](#运行评估工具)

# 安装
运行Megatron-LM和Megatron-Core推荐使用英伟达提供的官方镜像 nvcr.io/nvidia/pytorch:23.12-py3 来创建容器

```bash
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
cd Pai-Megatron-Patch
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

运行MegaBlocks-MoE推荐使用英伟达提供的官方镜像 nvcr.io/nvidia/pytorch:23.09-py3 来创建容器
```bash
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
cd Pai-Megatron-Patch
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

git clone https://github.com/databricks/megablocks
cd megablocks
pip install -e .
pip install transformers==4.38.2
```

# 数据集和模型下载
```bash
cd /mnt
mkdir qwen-ckpts
cd qwen-ckpts
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-ckpts/Qwen1.5-0.5B.tgz
tar -zxf Qwen1.5-0.5B.tgz

mkdir qwen-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/wudao_qwenbpe_text_document.bin
wgt https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/wudao_qwenbpe_text_document.idx

wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/alpaca_zh-qwen-train.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/alpaca_zh-qwen-valid.json


```

# Megatron-LM-Dense模型训练流程

## Megatron-LM-Dense模型格式转换
使用我们提供的模型转换脚本，将huggingface格式的模型文件转换为megatron格式：
```bash
```

## Megatron-LM-Dense继续预训练
运行run_pretrain_megatron_llama.sh脚本，需要传入的参数列表如下
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
```

## Megatron-LM-Dense继续预训练
运行run_finetune_megatron_llama.sh脚本，需要传入的参数列表如下
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
```

## Megatron-LM-Dense指令微调

```bash
```

# Megatron-Core-Dense模型训练流程

## Megatron-Core-Dense模型格式转换

## Megatron-Core-Dense继续预训练

## Megatron-Core-Dense指令微调

# Megatron-Core-MoE模型训练流程

## Megatron-Core-MoE模型格式转换

## Megatron-Core-MoE继续预训练

## Megatron-Core-MoE指令微调

# MegaBlocks-MoE模型训练流程

## MegaBlocks-MoE模型格式转换
使用我们提供的模型转换脚本，将huggingface格式的模型文件转换为MegaBlocks-MoE格式：
```bash
cd /workspace/PAI-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen
sh hf2megablocks_convertor_1.5.sh \
/mnt/qwen-ckpts/Qwen1.5-0.5B \
../../.. \
/mnt/qwen-ckpts/Qwen1.5-0.5B \
/mnt/qwen-ckpts/Qwen1.5-0.5B_megablocks_tp1_pp1_ep8_exp8 \
2  \
1  \
293  \
8  \
4  \
false
```

## MegaBlocks-MoE继续预训练
运行run_pretrain_megablocks_qwen.sh脚本，需要传入的参数列表如下
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
DO=${15}                        # MegaBlocks不支持Distributed Optimizer，这里需设置为False
FL=${16}                        # 是否使用Flash Attention: true, false
SP=${17}                        # 是否使用序列并行: true, false
TE=${18}                        # MegaBlocks不支持TE，这里需设置为False
MoE=${19}                       # 是否开启MoE，主要MoE具体参数需要在脚本中设置
SAVE_INTERVAL=${20}             # 保存ckpt的间隔
DATASET_PATH=${21}              # 训练数据集路径
PRETRAIN_CHECKPOINT_PATH=${22}  # 预训练模型路径
TRAIN_TOKENS=${23}              # 训练token数
WARMUP_TOKENS=${24}             # 预热token数
OUTPUT_BASEPATH=${25}           # 训练输出文件路径
```
单机运行示例如下：
```bash
cd /workspace/PAI-Megatron-Patch/examples/qwen1.5
sh run_pretrain_megablocks_qwen.sh  \
dsw  \
../.. \
0.5B  \
1  \
8  \
1e-5 \
1e-6  \
2048  \
32768  \
293  \
fp16  \
1  \
1  \
sel  \
false  \
false  \
true  \
false  \
true  \
100  \
/mnt/qwen-datasets/wudao_qwenbpe_text_document  \
/mnt/qwen-ckpts/Qwen1.5-0.5B  \
10000000000  \
100000  \
/mnt/qwen-ckpts/debug
```

## MegaBlocks-MoE指令微调

# 下游任务评估

## Megatron-LM-Dense模型转成Huggingface格式

## Megatron-Core-Dense模型转成Huggingface格式

## 运行评估工具