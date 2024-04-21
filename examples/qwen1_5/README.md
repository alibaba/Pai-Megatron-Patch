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

cd Pai-Megatron-Patch/Megatron-LM-MegaBlocks/megablocks
pip install -e .
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
运行hf2megatron_convertor.sh脚本，需要传入的参数列表如下
```
MEGATRON_PATH=$1                    # Megatraon-LM所在目录
SOURCE_CKPT_PATH=$2                 # 源ckpt的目录
TARGET_CKPT_PATH=$3                 # 目标ckpt的目录
TP=$4                               # 模型并行度
PP=$5                               # 流水并行度
MN=$6                               # 模型名称：qwen-7b,qwen-14b,qwen-72b;qwen1.5-0.5b,qwen1.5-1.8b,qwen1.5-4b,qwen1.5-7b,qwen1.5-14b,qwen1.5-72b
EXTRA_VOCAB_SIZE=$7                 # 额外扩充词表数
mg2hf=$8                            # 是否执行mg2hf转换
```

运行run_pretrain_megatron_qwen.sh脚本，需要传入的参数列表如下
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

运行run_finetune_megatron_qwen_withGA.sh脚本，需要传入的参数列表如下
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
VALID_DATASET_PATH=${21}        # 验证数据集路径
PRETRAIN_CHECKPOINT_PATH=${22}  # 预训练模型路径
TRAIN_ITERS=${23}               # 训练step数
WARMUP_ITERS=${24}              # 预热step数
OUTPUT_BASEPATH=${25}           # 训练输出文件路径
```

## Megatron-LM-Dense模型格式转换
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen
sh hf2megatron_convertor.sh \
../../../     \
/mnt/qwen-ckpts/Qwen1.5-0.5B    \
/mnt/qwen-ckpts/Qwen1.5-0.5B-hf-to-megatron-tp1-pp1  \
1  \
1  \
qwen1.5-0.5b \
0 \
false
```

## Megatron-LM-Dense继续预训练
```bash
cd /workspace/Pai-Megatron-Patch/examples/qwen1_5
sh run_pretrain_megatron_qwen.sh  \
dsw  \
../../ \
0.5B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
293   \
bf16  \
1   \
1  \
sel  \
true   \
false  \
false   \
false   \
100000  \
/mnt/qwen-datasets/wudao_qwenbpe_text_document  \
/mnt/qwen-ckpts/Qwen1.5-0.5B-hf-to-megatron-tp1-pp1  \
100000000   \
10000   \
/mnt/output_megatron_qwen
```

## Megatron-LM-Dense指令微调
```bash
cd /workspace/Pai-Megatron-Patch/examples/qwen1_5
sh run_finetune_megatron_qwen_withGA.sh  \
dsw  \
../../ \
0.5B     \
1      \
32     \
1e-5   \
1e-6   \
128   \
128     \
293      \
bf16   \
1      \
1      \
sel    \
true   \
false  \
false  \
false \
100 \
/mnt/qwen-datasets/alpaca_zh-qwen-train.json   \
/mnt/qwen-datasets/alpaca_zh-qwen-valid.json   \
/mnt/qwen-ckpts/Qwen1.5-0.5B-hf-to-megatron-tp1-pp1  \
1000 \
10 \
/mnt/output_megatron_qwen
```

# Megatron-Core-Dense模型训练流程
运行hf2mcore_qwen1.5_convertor.sh脚本，需要传入的参数列表如下
```
MODEL_SIZE=$1                  # 模型参数：0.5B/1.8B
HG_CKPT_PATH=$2                # HF的CKPT的路径
MEGATRON_PATH=$3               # Megatron-LM的根目录
SOURCE_CKPT_PATH=$4            # 源路径
TARGET_CKPT_PATH=$5            # 目标路径
TP=$6                          # 模型并行度
PP=$7                          # 流水并行度
EXTRA_VOCAB_SIZE=$8            # 额外扩充词表大小
NUM_EXPERTS=$9                 # 专家数量
EXPERTS_TOPK=${10}             # 专家路由Topk
EP=${11}                       # 专家并行度
mg2hf=${12}                    # 是否执行mcore2hf转换
```

运行run_pretrain_mcore_qwen.sh脚本，需要传入的参数列表如下
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
MOE=${19}                       # 是否打开MOE: true, false
SAVE_INTERVAL=${20}             # 保存ckpt的间隔
DATASET_PATH=${21}              # 训练数据集路径
PRETRAIN_CHECKPOINT_PATH=${22}  # 预训练模型路径
TRAIN_TOKENS=${23}              # 训练token数
WARMUP_TOKENS=${24}             # 预热token数
OUTPUT_BASEPATH=${25}           # 训练输出文件路径
```

运行run_finetune_mcore_qwen_withGA.sh脚本，需要传入的参数列表如下
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
MOE=${19}                       # 是否打开MOE: true, false
SAVE_INTERVAL=${20}             # 保存ckpt的间隔
DATASET_PATH=${21}              # 训练数据集路径
VALID_DATASET_PATH=${22}        # 验证数据集路径
PRETRAIN_CHECKPOINT_PATH=${23}  # 预训练模型路径
TRAIN_ITERS=${24}               # 训练step数
WARMUP_ITERS=${25}              # 预热step数
OUTPUT_BASEPATH=${26}           # 训练输出文件路径
```

## Megatron-Core-Dense模型格式转换
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen \
sh hf2mcore_qwen1.5_convertor.sh \
0.5B \
/mnt/qwen-ckpts/Qwen1.5-0.5B \
../../../     \
/mnt/qwen-ckpts/Qwen1.5-0.5B \
/mnt/qwen-ckpts/Qwen1.5-0.5B-hf-to-mcore-tp1-pp1  \
1  \
1  \
293  \
0  \
0  \
0 \
false
```

## Megatron-Core-Dense继续预训练
```bash
cd /workspace/Pai-Megatron-Patch/examples/qwen1_5
sh run_pretrain_mcore_qwen.sh  \
dsw  \
../../ \
0.5B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
293   \
bf16  \
1   \
1  \
sel  \
true   \
false  \
false   \
false   \
false \
100000  \
/mnt/qwen-datasets/wudao_qwenbpe_text_document  \
/mnt/qwen-ckpts/Qwen1.5-0.5B-hf-to-mcore-tp1-pp1   \
100000000   \
10000   \
/mnt/output_mcore_qwen
```

## Megatron-Core-Dense指令微调
```bash
cd /workspace/Pai-Megatron-Patch/examples/qwen1_5
sh run_finetune_mcore_qwen_withGA.sh  \
dsw  \
../../ \
0.5B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
293   \
bf16  \
1   \
1  \
sel  \
true   \
false  \
false   \
false   \
false \
100000  \
/mnt/qwen-datasets/alpaca_zh-qwen-train.json   \
/mnt/qwen-datasets/alpaca_zh-qwen-valid.json   \
/mnt/qwen-ckpts/Qwen1.5-0.5B-hf-to-mcore-tp1-pp1   \
100000000   \
10000   \
/mnt/output_mcore_qwen
```

# Megatron-Core-MoE模型训练流程

## Megatron-Core-MoE模型格式转换
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen \
sh hf2mcore_qwen1.5_convertor.sh \
0.5B \
/mnt/qwen-ckpts/Qwen1.5-0.5B \
../../../     \
/mnt/qwen-ckpts/Qwen1.5-0.5B \
/mnt/qwen-ckpts/Qwen1.5-0.5B-hf-to-mcore-tp1-pp1-ep1-exp8 \
1  \
1  \
293  \
8  \
2  \
1 \
false
```

## Megatron-Core-MoE继续预训练
```bash
cd /workspace/Pai-Megatron-Patch/examples/qwen1_5
sh run_pretrain_mcore_qwen.sh  \
dsw  \
../../ \
0.5B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
293   \
bf16  \
1   \
1  \
sel  \
true   \
false  \
false   \
false   \
true \
100000  \
/mnt/qwen-datasets/wudao_qwenbpe_text_document  \
/mnt/qwen-ckpts/Qwen1.5-0.5B-hf-to-mcore-tp1-pp1-ep1-exp8   \
100000000   \
10000   \
/mnt/output_mcore_qwen
```

## Megatron-Core-MoE指令微调
```bash
cd /workspace/Pai-Megatron-Patch/examples/qwen1_5
sh run_finetune_mcore_qwen_withGA.sh  \
dsw  \
../../ \
0.5B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
293   \
bf16  \
1   \
1  \
sel  \
true   \
false  \
false   \
false   \
true \
100000  \
/mnt/qwen-datasets/alpaca_zh-qwen-train.json   \
/mnt/qwen-datasets/alpaca_zh-qwen-valid.json   \
/mnt/qwen-ckpts/Qwen1.5-0.5B-hf-to-mcore-tp1-pp1-ep1-exp8   \
100000000   \
10000   \
/mnt/output_mcore_qwen
```

# MegaBlocks-MoE模型训练流程

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

运行run_finetune_megablocks_qwen_withGA.sh脚本，需要传入的参数列表如下
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
VALID_DATASET_PATH=${22}        # 验证数据集路径
PRETRAIN_CHECKPOINT_PATH=${23}  # 预训练模型路径
TRAIN_ITERS=${24}               # 训练step数
WARMUP_ITERS=${25}              # 预热step数
OUTPUT_BASEPATH=${26}           # 训练输出文件路径
```

## MegaBlocks-MoE模型格式转换
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen
sh hf2megablocks_qwen1.5_convertor.sh \
0.5B  \
/mnt/qwen-ckpts/Qwen1.5-0.5B \
../../.. \
/mnt/qwen-ckpts/Qwen1.5-0.5B \
/mnt/qwen-ckpts/Qwen1.5-0.5B-hf-to-megablocks-tp1-pp1-ep8-exp8 \
1  \
1  \
293  \
8  \
2 \
8 \
false
```

## MegaBlocks-MoE继续预训练
```bash
cd /workspace/Pai-Megatron-Patch/examples/qwen1_5
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
/mnt/qwen-ckpts/Qwen1.5-0.5B-hf-to-megablocks-tp1-pp1-ep8-exp8 \
10000000000  \
100000  \
/mnt/output_mcore_qwen
```

## MegaBlocks-MoE指令微调
```bash
cd /workspace/Pai-Megatron-Patch/examples/qwen1_5
sh run_finetune_megablocks_qwen_withGA.sh  \
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
/mnt/qwen-datasets/alpaca_zh-qwen-train.json   \
/mnt/qwen-datasets/alpaca_zh-qwen-valid.json   \
/mnt/qwen-ckpts/Qwen1.5-0.5B-hf-to-megablocks-tp1-pp1-ep8-exp8 \
10000000000  \
100000  \
/mnt/output_mcore_qwen
```

# 下游任务评估

## Megatron-LM-Dense模型转成Huggingface格式
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen
sh hf2megatron_convertor.sh \
../../../     \
/mnt/qwen-ckpts/Qwen1.5-0.5B-hf-to-megatron-tp1-pp1/release  \
/mnt/qwen-ckpts/Qwen1.5-0.5B-megatron-to-hf    \
1  \
1  \
qwen1.5-0.5b \
0 \
true
```

## 运行评估工具
```bash
cd /workspace/Pai-Megatron-Patch/LM-Evaluation-Harness-240310
accelerate launch --main_process_port 29051 -m lm_eval \
--model hf \
--model_args pretrained=/mnt/qwen-ckpts/Qwen1.5-0.5B-megatron-to-hf,trust_remote_code=True \
--tasks mmlu,ceval-valid  \
--batch_size 16
```