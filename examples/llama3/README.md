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
   * [下游任务评估](#下游任务评估)
      * [Megatron-LM模型格式转换](#Megatron-LM-Dense模型转成Huggingface格式)
      * [运行评估工具](#运行评估工具)

> Pai-Megatron-Patch 目前已支持[LLaMA3.1](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/llama3_1/README.md), 
> 推荐您使用LLaMA3.1 代替 LLaMA3 以享受Pai-Megatron-Patch中最新集成的优化技术。

# 安装
推荐使用英伟达提供的官方镜像 nvcr.io/nvidia/pytorch:23.12-py3 来创建容器

```bash
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
cd Pai-Megatron-Patch
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

# 数据集和模型下载
```bash
cd /mnt
mkdir llama3-ckpts
cd llama3-ckpts
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-ckpts/Meta-Llama-3-8B.tgz
tar -zxf Meta-Llama-3-8B.tgz

mkdir llama3-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/wudao_llama3bpe_content_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/wudao_llama3bpe_content_document.idx

wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/alpaca_zh-llama3-train.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/alpaca_zh-llama3-valid.json
```

# Megatron-LM-Dense模型训练流程

运行hf2megatron_convertor.sh脚本，需要传入的参数列表如下
```
MEGATRON_PATH=$1                   # Megatron-LM的路径
SOURCE_CKPT_PATH=$2                # 原始CKPT的路径
TARGET_CKPT_PATH=$3                # 目标CKPT的路径
TP=$4                              # 模型并行度
PP=$5                              # 流水并行度
MN=$6                              # llama3-8b 
EXTRA_VOCAB_SIZE=$7                # 词表扩充大小
mg2hf=$8                           # 是否执行mg2hf转换
```

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

运行run_finetune_megatron_llama_withGA.sh脚本，需要传入的参数列表如下
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
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/llama
sh hf2megatron_convertor.sh \
../../../     \
/mnt/llama3-ckpts/Meta-Llama-3-8B    \
/mnt/llama3-ckpts/Meta-Llama-3-8B-to-megatron-tp4-pp1  \
4  \
1  \
llama3-8b \
0 \
false
```

## Megatron-LM-Dense继续预训练
```bash
cd /workspace/Pai-Megatron-Patch/examples/llama3
sh run_pretrain_megatron_llama.sh  \
dsw  \
../../ \
8B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
256   \
bf16  \
4   \
1  \
sel  \
true   \
false  \
false   \
false   \
100000  \
/mnt/llama3-datasets/wudao_llama3bpe_content_document  \
/mnt/llama3-ckpts/Meta-Llama-3-8B-to-megatron-tp4-pp1  \
100000000   \
10000   \
/mnt/output_megatron_llama3
```

## Megatron-LM-Dense指令微调
```bash
cd /workspace/Pai-Megatron-Patch/examples/llama3
sh run_finetune_megatron_llama_withGA.sh  \
dsw  \
../../ \
8B     \
1      \
32     \
1e-5   \
1e-6   \
128   \
128     \
256      \
bf16   \
4      \
1      \
sel    \
true   \
false  \
false  \
false \
100 \
/mnt/llama3-datasets/alpaca_zh-llama2-train.json   \
/mnt/llama3-datasets/alpaca_zh-llama2-valid.json   \
/mnt/llama3-ckpts/Meta-Llama-3-8B-to-megatron-tp4-pp1  \
1000 \
10 \
/mnt/output_megatron_llama3/
```

# Megatron-Core-Dense模型训练流程

运行hf2mcore_convertor.sh脚本，需要传入的参数列表如下
```
MODEL_SIZE=$1                  # 模型参数：7B/13B/70B
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

运行run_pretrain_mcore_llama.sh脚本，需要传入的参数列表如下
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

运行run_finetune_mcore_llama_withGA.sh脚本，需要传入的参数列表如下
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
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/llama \
sh hf2mcore_convertor.sh \
8B \
/mnt/llama3-ckpts/Meta-Llama-3-8B    \
../../../     \
/mnt/llama3-ckpts/Meta-Llama-3-8B    \
/mnt/llama3-ckpts/Meta-Llama-3-8B-to-mcore-tp4-pp1  \
4  \
1  \
256  \
0  \
0  \
0 \
false
```

## Megatron-Core-Dense继续预训练
```bash
cd /workspace/Pai-Megatron-Patch/examples/llama3
sh run_pretrain_mcore_llama.sh  \
dsw  \
../../ \
8B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
256   \
bf16  \
4   \
1  \
sel  \
true   \
false  \
false   \
false   \
false \
100000  \
/mnt/llama3-datasets/wudao_llama3bpe_content_document   \
/mnt/llama3-ckpts/Meta-Llama-3-8B-to-mcore-tp4-pp1  \
100000000   \
10000   \
/mnt/output_mcore_llama3
```

## Megatron-Core-Dense指令微调
```bash
cd /workspace/Pai-Megatron-Patch/examples/llama3
sh run_finetune_mcore_llama_withGA.sh  \
dsw  \
../../ \
8B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
256   \
bf16  \
4   \
1  \
sel  \
true   \
false  \
false   \
false   \
false \
100000  \
/mnt/llama3-datasets/alpaca_zh-llama3-train.json   \
/mnt/llama3-datasets/alpaca_zh-llama3-valid.json   \
/mnt/llama3-ckpts/Meta-Llama-3-8B-to-mcore-tp4-pp1  \
100000000   \
10000   \
/mnt/output_mcore_llama3
```

# 下游任务评估

## Megatron-LM-Dense模型转成Huggingface格式
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/llama
sh hf2megatron_convertor.sh \
../../../     \
/mnt/llama3-ckpts/Meta-Llama-3-8B-to-mcore-tp4-pp1/release  \
/mnt/llama3-ckpts/Meta-Llama-3-8B-hf-megatron-to-hf    \
4  \
1  \
llama3-8b \
0 \
true
```

请将开源Huggingface模型文件夹路径下的.json (pytorch_model.bin.index.json除外)文件拷贝至/mnt/llama3-ckpts/Meta-Llama-3-8B-hf-megatron-to-hf目录下，以保证模型可以正常使用。

## 运行评估工具
```bash
cd /workspace/Pai-Megatron-Patch/LM-Evaluation-Harness-240310
accelerate launch --main_process_port 29051 -m lm_eval \
--model hf \
--model_args pretrained=/mnt/llama3-ckpts/Meta-Llama-3-8B-hf-megatron-to-hf,trust_remote_code=True \
--tasks mmlu,ceval-valid  \
--batch_size 16
```