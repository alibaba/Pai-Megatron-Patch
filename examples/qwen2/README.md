# Table of Contents
   * [安装](#安装)
   * [数据集&模型下载](#数据集和模型下载)
   * [Megatron-Core-Dense模型训练流程](#Megatron-Core-Dense模型训练流程)
      * [模型格式转换](#Megatron-Core-Dense模型格式转换)
      * [模型评估验证](#Megatron-Core-Dense模型评估验证)
      * [继续预训练](#Megatron-Core-Dense继续预训练)
      * [指令微调](#Megatron-Core-Dense指令微调)
   * [Megatron-Core-Moe模型训练流程](#Megatron-Core-MoE模型训练流程)
      * [模型格式转换](#Megatron-Core-MoE模型格式转换)
      * [继续预训练](#Megatron-Core-MoE继续预训练)
      * [指令微调](#Megatron-Core-MoE指令微调)
   * [下游任务评估](#下游任务评估)
      * [Megatron-Core模型格式转换](#Megatron-Core-Dense模型转成Huggingface格式)
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
mkdir qwen-ckpts
cd qwen-ckpts
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-ckpts/Qwen2-0.5B.tgz
tar -zxf Qwen2-0.5B.tgz

mkdir qwen-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/wudao_qwenbpe_text_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/wudao_qwenbpe_text_document.idx

wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/alpaca_zh-qwen-train.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/alpaca_zh-qwen-valid.json


```


# Megatron-Core-Dense模型训练流程
dense模式的格式转换命令如下，运行hf2mcore_qwen2_convertor.sh脚本，需要传入的参数列表如下
```
MODEL_SIZE=$1                  # 模型参数：0.5B/1.8B
SOURCE_CKPT_PATH=$2            # 源路径
TARGET_CKPT_PATH=$3            # 目标路径
TP=$4                          # 模型并行度
PP=$5                          # 流水并行度
EP=$6                          # 专家并行度
PR=$7                          # 转换精度
USE_TE=$8                      # 是否使用Transformer Engine建模
mg2hf=$9                       # 是否执行mcore2hf转换
HG_CKPT_PATH=${10}                # HF的CKPT的路径
```


运行run_pretrain_qwen.sh脚本，需要传入的参数列表如下
```
ENV=$1                          # 运行环境: dlc, dsw
MODEL_SIZE=$2                   # 模型结构参数量级：7B, 72B
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
TE=${17}                        # 是否使用Transformer Engine: true, false
SAVE_INTERVAL=${18}             # 保存ckpt的间隔
DATASET_PATH=${19}              # 训练数据集路径
PRETRAIN_CHECKPOINT_PATH=${20}  # 预训练模型路径
TRAIN_TOKENS=${21}              # 训练token数
WARMUP_TOKENS=${22}             # 预热token数
OUTPUT_BASEPATH=${23}           # 训练输出文件路径
```

运行run_finetune_qwen.sh脚本，需要传入的参数列表如下
```
ENV=$1                          # 运行环境: dlc, dsw
MODEL_SIZE=$2                   # 模型结构参数量级：7B, 72B
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
TE=${17}                        # 是否使用Transformer Engine: true, false
SAVE_INTERVAL=${18}             # 保存ckpt的间隔
DATASET_PATH=${19}              # 训练数据集路径
VALID_DATASET_PATH=${20}        # 验证数据集路径
PRETRAIN_CHECKPOINT_PATH=${21}  # 预训练模型路径
TRAIN_ITERS=${22}               # 训练step数
WARMUP_ITERS=${23}              # 预热step数
OUTPUT_BASEPATH=${24}           # 训练输出文件路径
```

## Megatron-Core-Dense模型格式转换
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen \
sh hf2mcore_qwen2_convertor.sh \
0.5B \
/mnt/qwen-ckpts/Qwen2-0.5B \
/mnt/qwen-ckpts/Qwen2-0.5B-hf-to-mcore-te-tp1-pp1  \
1  \
1  \
1 \
fp32 \
true \
false 
```

## Megatron-Core-Dense模型评估验证

Huggingface模型计算loss
```bash
cd /workspace/Pai-Megatron-Patch/examples/qwen2 \
sh run_evaluate_huggingface_qwen.sh \
0.5B \
1 \
256 \
256 \
bf16 \
/mnt/qwen-datasets/alpaca_zh-qwen-valid.json \
/mnt/qwen-ckpts/Qwen2-0.5B
```

Mcore模型计算loss
```bash
cd /workspace/Pai-Megatron-Patch/examples/qwen2 \
sh run_evaluate_mcore_qwen.sh \
0.5B \
1 \
256 \
256 \
bf16 \
1 \
1 \
sel \
true \
false \
false \
true \
/mnt/qwen-datasets/alpaca_zh-qwen-valid.json \
/mnt/qwen-ckpts/Qwen2-0.5B-hf-to-mcore-te-tp1-pp1

```

## Megatron-Core-Dense继续预训练
```bash
cd /workspace/Pai-Megatron-Patch/examples/qwen2 \
sh run_pretrain_qwen.sh  \
dsw  \
0.5B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
bf16  \
1   \
1  \
1 \
sel  \
true   \
false  \
false   \
true   \
100000  \
/mnt/qwen-datasets/wudao_qwenbpe_text_document  \
/mnt/qwen-ckpts/Qwen2-0.5B-hf-to-mcore-te-tp1-pp1   \
100000000   \
10000   \
/mnt/output_mcore_qwen
```

## Megatron-Core-Dense指令微调
```bash
cd /workspace/Pai-Megatron-Patch/examples/qwen2 \
sh run_finetune_qwen.sh  \
dsw  \
0.5B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
bf16  \
1   \
1  \
1 \
sel  \
true   \
false  \
false   \
true   \
100000  \
/mnt/qwen-datasets/alpaca_zh-qwen-train.json   \
/mnt/qwen-datasets/alpaca_zh-qwen-valid.json   \
/mnt/qwen-ckpts/Qwen2-0.5B-hf-to-mcore-te-tp1-pp1   \
100000000   \
10000   \
/mnt/output_mcore_qwen
```

# Megatron-Core-MoE模型训练流程

## Megatron-Core-MoE模型格式转换
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen \
sh hf2mcore_qwen2_convertor.sh \
A14B \
/mnt/qwen-ckpts/Qwen2-57B-A14B \
/mnt/qwen-ckpts/Qwen2-57B-A14B-hf-to-mcore-te-tp4-pp1-ep4  \
4  \
1  \
4 \
fp32 \
true \
false 
```

## Megatron-Core-MoE继续预训练
```bash
cd /workspace/Pai-Megatron-Patch/examples/qwen2 \
sh run_pretrain_qwen.sh  \
dsw  \
A14B  \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
bf16  \
4   \
1  \
4 \
sel  \
true   \
false  \
false   \
true   \
100000  \
/mnt/qwen-datasets/wudao_qwenbpe_text_document  \
/mnt/qwen-ckpts/Qwen2-57B-A14B-hf-to-mcore-te-tp4-pp1-ep4   \
100000000   \
10000   \
/mnt/output_mcore_qwen
```

## Megatron-Core-MoE指令微调
```bash
cd /workspace/Pai-Megatron-Patch/examples/qwen2 \
sh run_finetune_qwen.sh  \
dsw  \
A14B  \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
bf16  \
4   \
1  \
4 \
sel  \
true   \
false  \
false   \
true   \
100000  \
/mnt/qwen-datasets/alpaca_zh-qwen-train.json   \
/mnt/qwen-datasets/alpaca_zh-qwen-valid.json   \
/mnt/qwen-ckpts/Qwen2-57B-A14B-hf-to-mcore-te-tp4-pp1-ep4   \
100000000   \
10000   \
/mnt/output_mcore_qwen
```



# 下游任务评估

## Megatron-Core-Dense模型转成Huggingface格式
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen
bash hf2mcore_qwen2_convertor.sh \
0.5B \
/mnt/qwen-ckpts/Qwen2-0.5B-hf-to-mcore-te-tp1-pp1  \
/mnt/qwen-ckpts/Qwen2-0.5B-mcore-te-to-hf    \
1  \
1  \
1 \
fp32 \
true \
true \
/mnt/qwen-ckpts/Qwen2-0.5B
```

## 运行评估工具
```bash
cd /workspace/Pai-Megatron-Patch/LM-Evaluation-Harness-240310
accelerate launch --main_process_port 29051 -m lm_eval \
--model hf \
--model_args pretrained=/mnt/qwen-ckpts/Qwen2-0.5B-mcore-te-to-hf,trust_remote_code=True \
--tasks mmlu,ceval-valid  \
--batch_size 16
```