# Table of Contents
   * [安装](#安装)
   * [数据集&模型下载](#数据集和模型下载)
   * [Megatron-Core-MoE模型训练流程](#Megatron-Core-MoE模型训练流程)
      * [模型格式转换](#Megatron-Core-MoE模型格式转换)
      * [继续预训练](#Megatron-Core-MoE继续预训练)
      * [指令微调](#Megatron-Core-MoE指令微调)
   * [下游任务评估](#下游任务评估)
      * [Megatron-Core模型格式转换](#Megatron-Core-MoE模型转成Huggingface格式)
      * [运行评估工具](#运行评估工具)

# 安装
运行Megatron-LM和Megatron-Core推荐使用英伟达提供的官方镜像 nvcr.io/nvidia/pytorch:23.12-py3 来创建容器

```bash
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
cd Pai-Megatron-Patch
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

# 数据集和模型下载
```bash
cd /mnt
mkdir deepseek-ckpts
cd deepseek-ckpts
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-ckpts/DeepSeek-V2-Lite.tgz
tar -zxf DeepSeek-V2-Lite.tgz

mkdir deepseek-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/SlimPajama.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/alpaca_zh-train.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/alpaca_zh-valid.json

cd /workspace/Pai-Megatron-Patch/toolkits/pretrain_data_preprocessing
sh run_make_pretraining_dataset_megatron.sh \
/mnt/deepseek-datasets/SlimPajama.json \
DeepSeekV2Tokenizer \
text \
/mnt/deepseek-datasets/ \
/mnt/deepseek-ckpts/DeepSeek-V2-Lite

```

# Megatron-Core-MoE模型训练流程

MoE模型的格式转换命令如下，运行hf2mcore_deepseek_v2_moe_convertor.sh脚本，需要传入的参数列表如下
```
MODEL_SIZE=$1                  # 模型参数：0.5B/1.8B
SOURCE_CKPT_PATH=$2            # 源路径
TARGET_CKPT_PATH=$3            # 目标路径
TP=$4                          # 模型并行度
PP=$5                          # 流水并行度
EP=$6                          # 专家并行度
mg2hf=$6                       # 是否执行mcore2hf转换
HG_CKPT_PATH=$7                # HF的CKPT的路径
```

运行run_pretrain_deepseek.sh脚本，需要传入的参数列表如下
```
ENV=$1                          # 运行环境: dlc, dsw
MODEL_SIZE=$2                   # 模型结构参数量级：A21B, A2.4B 
BATCH_SIZE=$3                   # 每卡训练一次迭代样本数: 4, 8
GLOBAL_BATCH_SIZE=$4            # 全局batch size
LR=$5                           # 学习率: 1e-5, 5e-5
MIN_LR=$6                       # 最小学习率: 1e-6, 5e-6
SEQ_LEN=$7                      # 序列长度
PAD_LEN=$8                      # Padding长度：100
PR=$9                       # 训练精度: fp16, bf16
TP=${10}                        # 模型并行度
PP=${11}                        # 流水并行度
EP=${12}                        # 专家并行度
AC=${13}                        # 激活检查点模式: sel, full
DO=${14}                        # 是否使用Megatron版Zero-1降显存优化器: true, false
FL=${15}                        # 是否使用Flash Attention: true, false
SP=${16}                        # 是否使用序列并行: true, false
SAVE_INTERVAL=${17}             # 保存ckpt的间隔
DATASET_PATH=${18}              # 训练数据集路径
PRETRAIN_CHECKPOINT_PATH=${19}  # 预训练模型路径
TRAIN_TOKENS=${20}              # 训练token数
WARMUP_TOKENS=${21}             # 预热token数
OUTPUT_BASEPATH=${22}           # 训练输出文件路径
```

运行run_finetune_deepseek.sh脚本，需要传入的参数列表如下
```
ENV=$1                          # 运行环境: dlc, dsw
MODEL_SIZE=$2                   # 模型结构参数量级：7B, 14B
BATCH_SIZE=$3                   # 每卡训练一次迭代样本数: 4, 8
GLOBAL_BATCH_SIZE=$4            # 全局batch size
LR=$5                           # 学习率: 1e-5, 5e-5
MIN_LR=$6                       # 最小学习率: 1e-6, 5e-6
SEQ_LEN=$7                      # 序列长度
PAD_LEN=$8                      # Padding长度：100
PR=$9                           # 训练精度: fp16, bf16
TP=${10}                        # 模型并行度
PP=${11}                        # 流水并行度
EP=${12}                        # 专家并行度
AC=${13}                        # 激活检查点模式: sel, full
DO=${14}                        # 是否使用Megatron版Zero-1降显存优化器: true, false
FL=${15}                        # 是否使用Flash Attention: true, false
SP=${16}                        # 是否使用序列并行: true, false
SAVE_INTERVAL=${17}             # 保存ckpt的间隔
DATASET_PATH=${18}              # 训练数据集路径
VALID_DATASET_PATH=${19}        # 验证数据集路径
PRETRAIN_CHECKPOINT_PATH=${20}  # 预训练模型路径
TRAIN_ITERS=${21}               # 训练step数
WARMUP_ITERS=${22}              # 预热step数
OUTPUT_BASEPATH=${23}           # 训练输出文件路径
```

## Megatron-Core-MoE模型格式转换

```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/deepseek \
bash hf2mcore_deepseek_v2_moe_convertor.sh \
A2.4B \
/mnt/deepseek-ckpts/DeepSeek-V2-Lite \
/mnt/deepseek-ckpts/DeepSeek-V2-Lite-to-mcore-tp1-pp1-ep4 \
1 \
1 \
4 \
false
```

## Megatron-Core-MoE继续预训练
```bash
cd /workspace/Pai-Megatron-Patch/examples/deepseek_v2
sh run_pretrain_deepseek.sh \
dsw \
A2.4B \
1 \
8 \
1e-5 \
1e-6 \
1024 \
1024 \
bf16 \
1 \
1 \
4 \
sel \
true \
true \
true \
500 \
/mnt/deepseek-datasets/mmap_deepseekv2_datasets_text_document  \
/mnt/deepseek-ckpts/DeepSeek-V2-Lite-to-mcore-tp1-pp1-ep4 \
100000000   \
10000   \
/mnt/deepseek-ckpts/test_pretrain
```

## Megatron-Core-MoE指令微调
```bash
cd /workspace/Pai-Megatron-Patch/examples/deepseek_v2
sh run_finetune_deepseek.sh  \
dsw \
A2.4B \
1    \
8    \
1e-5   \
1e-6   \
128  \
128  \
bf16  \
1   \
1  \
4 \
sel \
true \
true \
true \
100  \
/mnt/deepseek-datasets/alpaca_zh-train.json   \
/mnt/deepseek-datasets/alpaca_zh-valid.json   \
/mnt/deepseek-ckpts/DeepSeek-V2-Lite-to-mcore-tp1-pp1-ep4 \
100000   \
10000   \
/mnt/deepseek-ckpts/test_ft
```


# 下游任务评估

## Megatron-Core-MoE模型转成Huggingface格式
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/deepseek
bash hf2mcore_deepseek_v2_moe_convertor.sh \
A2.4B \
/mnt/deepseek-ckpts/DeepSeek-V2-Lite-to-mcore-tp1-pp1-ep4  \
/mnt/deepseek-ckpts/DeepSeek-V2-Lite-mcore-to-hf   \
1  \
1  \
4 \
true \
/mnt/deepseek-ckpts/DeepSeek-V2-Lite
```

## 运行评估工具
```bash
cd /workspace/Pai-Megatron-Patch/LM-Evaluation-Harness-240310
accelerate launch --main_process_port 29051 -m lm_eval \
--model hf \
--model_args pretrained=/mnt/deepseek-ckpts/DeepSeek-V2-Lite-mcore-to-hf,trust_remote_code=True \
--tasks mmlu,ceval-valid  \
--batch_size 16
```