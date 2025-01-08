# Mistral模型在Pai-Megatron-Patch的最佳实践

## Table of Contents
   * [安装](#安装)
   * [数据集&模型下载](#数据集和模型下载)
   * [Megatron-LM-Dense模型训练流程](#Megatron-LM-Dense模型训练流程)
      * [模型格式转换](#Megatron-LM-Dense模型格式转换)
      * [继续预训练](#Megatron-LM-Dense继续预训练)
      * [指令微调](#Megatron-LM-Dense指令微调)
   * [Megatron-Core模型训练流程](#Megatron-Core模型训练流程)
      * [模型格式转换](#Megatron-Core模型格式转换)
      * [继续预训练](#Megatron-Core继续预训练)
      * [指令微调](#Megatron-Core指令微调)
   * [下游任务评估](#下游任务评估)
      * [Megatron-LM模型格式转换](#Megatron-LM-Dense模型转成Huggingface格式)
      * [运行评估工具](#运行评估工具)

## 安装
请在阿里云人工智能平台PAI产品中填写专属镜像地址： `dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/pai-megatron-patch:24.07` 

运行下列代码克隆Pai-Megatron-Patch
```bash
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
cd Pai-Megatron-Patch
```

目前Mcore版本的Mistral已支持使用FlashAttention-3加速计算，但只能在Hopper架构的GPU卡上进行运算。若需要在H卡上使用FA3，请在DSW的容器中按如下指令安装并保存镜像
```bash
pip install "git+https://github.com/Dao-AILab/flash-attention.git#egg=flashattn-hopper&subdirectory=hopper"
python_path=`python -c "import site; print(site.getsitepackages()[0])"`
mkdir -p $python_path/flashattn_hopper
wget -P $python_path/flashattn_hopper https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/hopper/flash_attn_interface.py
```

## 预训练数据集和模型下载
```bash
cd /mnt
mkdir mistral-ckpts
cd mistral-ckpts
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-ckpts/Mistral-7B-v0.1.tgz
tar -zxf Mistral-7B-v0.1.tgz

mkdir mistral-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/wudao_mistralbpe_content_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/wudao_mistralbpe_content_document.idx

wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/alpaca_zh-mistral-train.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/alpaca_zh-mistral-valid.json

```

## Megatron-LM-Dense模型训练流程

运行hf2megatron_convertor.sh脚本，需要传入的参数列表如下
```
MEGATRON_PATH=$1                   # Megatron-LM的路径
SOURCE_CKPT_PATH=$2                # 原始CKPT的路径
TARGET_CKPT_PATH=$3                # 目标CKPT的路径
TP=$4                              # 模型并行度
PP=$5                              # 流水并行度
MN=$6                              # mistral-7b
EXTRA_VOCAB_SIZE=$7                # 词表扩充大小
mg2hf=$8                           # 是否执行mg2hf转换
```

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

运行run_finetune_megatron_mistral_withGA.sh脚本，需要传入的参数列表如下

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

### Megatron-LM-Dense模型格式转换
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/mistral
sh hf2megatron_convertor.sh \
../../../     \
/mnt/mistral-ckpts/Mistral-7B-v0.1    \
/mnt/mistral-ckpts/Mistral-7B-v0.1-hf-to-megatron-tp4-pp1  \
4  \
1  \
mistral-7b \
0 \
false
```

### Megatron-LM-Dense继续预训练
```bash
cd /workspace/Pai-Megatron-Patch/examples/mistral
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
/mnt/mistral-datasets/wudao_mistralbpe_content_document   \
/mnt/mistral-ckpts/Mistral-7B-v0.1-hf-to-megatron-tp4-pp1  \
100000000   \
10000   \
/mnt/output_megatron_mistral
```

### Megatron-LM-Dense指令微调
```bash
cd /workspace/Pai-Megatron-Patch/examples/mistral
sh run_finetune_megatron_mistral_withGA.sh  \
dsw  \
../../ \
7B     \
1      \
8      \
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
100000  \
/mnt/mistral-datasets/alpaca_zh-mistral-train.json   \
/mnt/mistral-datasets/alpaca_zh-mistral-valid.json   \
/mnt/mistral-ckpts/Mistral-7B-v0.1-hf-to-megatron-tp4-pp1   \
100000000   \
10000   \
/mnt/output_megatron_mistral
```

## Megatron-Core-Dense模型训练流程
### Megatron-Core模型格式转换
运行hf2mcore_convertor.sh脚本，需要传入的参数列表如下
```
MODEL_SIZE=$1                  # 模型参数：7B/8x7B
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
WS=${13}                       # 当8x7B时，指定world size
```
例如，使用下述脚本将checkpoint转换到MCore-Dense并检查输出
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/mistral \
sh hf2mcore_convertor.sh \
7B \
/mnt/mistral-ckpts/Mistral-7B-v0.1 \
../../../     \
/mnt/mistral-ckpts/Mistral-7B-v0.1 \
/mnt/mistral-ckpts/Mistral-7B-v0.1-to-mcore-tp4-pp1  \
4  \
1  \
0  \
0  \
0  \
0 \
false
```

基于Sparse-Upcycled的Dense转MoE模型格式转换
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/mistral \
sh hf2mcore_convertor.sh \
7B \
/mnt/mistral-ckpts/Mistral-7B-v0.1 \
../../../     \
/mnt/mistral-ckpts/Mistral-7B-v0.1 \
/mnt/mistral-ckpts/Mistral-7B-v0.1-to-mcore-tp4-pp1-ep1-exp2 \
4  \
1  \
0  \
2  \
2  \
1 \
false
```

直接对Mixtral-8x7B模型进行Mcore的格式转换
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/mistral \
sh hf2mcore_convertor.sh \
8x7B \
/mnt/mistral-ckpts/Mixtral-8x7B-v0.1 \
../../../     \
/mnt/mistral-ckpts/Mixtral-8x7B-v0.1 \
/mnt/mistral-ckpts/Mixtral-8x7B-v0.1-to-mcore-tp4-pp1-ep4-exp8-ws16 \
4  \
1  \
0  \
8  \
2  \
4 \
false \
16
```

### Megatron-Core预训练及指令微调
在Mcore版的Mistral中，我们已将预训练和微调整合到`run_mcore_mistral.sh`脚本，对于不同的使用场景，二者各参数的意义有所不同。

#### 预训练&微调命令统一描述
需要传入的参数列表如下：
```bash
ENV=$1                          # 运行环境配置开关: dsw单机训练训练，dlc表示多机训练环境
MODEL_SIZE=$2                   # 模型结构参数量级: 7B/8x7B
BATCH_SIZE=$3                   # 一次迭代一个数据并行内的样本数
GLOBAL_BATCH_SIZE=$4            # 一次迭代多个数据并行的总样本数
LR=$5                           # 学习率
MIN_LR=$6                       # 最小学习率
SEQ_LEN=$7                      # 序列长度
PAD_LEN=$8                      # Padding长度
PR=${9}                         # 训练精度: fp16, bf16, fp8
TP=${10}                        # 模型并行度
PP=${11}                        # 流水并行度
CP=${12}                        # 上下文并行度
EP=${13}                        # 专家并行度
SP=${14}                        # 是否使用序列并行: true, false
DO=${15}                        # 是否使用Megatron版Zero-1降显存优化器: true, false
FL=${16}                        # 是否优先使用Flash Attention: true, false
SFT=${17}                       # 是否执行微调训练: true, false
AC=${18}                        # 激活检查点模式: sel, full, offload, false
OPTIMIZER_OFFLOAD=${19}         # 是否启用Offload optimizer: false, static, auto
SAVE_INTERVAL=${20}             # 保存ckpt的间隔
DATASET_PATH=${21}              # 训练数据集路径
VALID_DATASET_PATH=${22}        # 验证数据集路径
PRETRAIN_CHECKPOINT_PATH=${23}  # 预训练模型路径
TRAIN_TOKENS_OR_ITERS=${24}     # 训练TOKEN或者Iter数
WARMUP_TOKENS_OR_ITERS=${25}    # 预热TOKEN或者Iter数        
OUTPUT_BASEPATH=${26}           # 训练输出日志文件路径
```

#### 预训练示例
使用以下命令启动对mixtral的继续预训练。
备注：当`AC=offload`或`full`时，可设置`MP_AC_LAYERS`环境变量来控制Checkpointing或Offload的TransformerLayer层数（默认值：`1`）。

```bash
cd /workspace/Pai-Megatron-Patch/examples/mistral
sh run_mcore_mistral.sh  \
dsw  \
7B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
bf16  \
4   \
1  \
1 \
1 \
true \
true   \
true \
false \
false   \
false \
100000  \
/mnt/mistral-datasets/wudao_mistralbpe_content_document   \
/mnt/mistral-datasets/wudao_mistralbpe_content_document   \
/mnt/mistral-ckpts/Mistral-7B-v0.1-to-mcore-tp4-pp1  \
10000  \
100   \
/workspace/output_mcore_mistral_pretrain
```

#### 指令微调示例
制作idxmap用于微调的数据集可以参考[链接](https://github.com/alibaba/Pai-Megatron-Patch/tree/main/toolkits/sft_data_preprocessing)。
当准备好微调数据集后，将SFT开关设置为`true`即可进行指令微调。

```bash
cd /workspace/Pai-Megatron-Patch/examples/mistral
sh run_mcore_qwen.sh  \
dsw  \
7B   \
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
1 \
true \
true   \
true \
true \
false   \
false \
100000  \
/mnt/qwen-datasets/path_to_your_dataset   \
/mnt/qwen-datasets/path_to_your_dataset   \
/path/to/pretraining/checkpoint  \
10000  \
100   \
/workspace/output_mcore_mistral_finetune
```
通过设置MP_DATASET_TYPE环境变量，本脚本还可使用json格式的数据集进行指令微调
```bash
export MP_DATASET_TYPE="raw"
cd /workspace/Pai-Megatron-Patch/examples/mistral
sh run_mcore_qwen.sh  \
dsw  \
7B   \
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
1 \
true \
true   \
true \
true \
false   \
false \
100000  \
/mnt/mistral-datasets/alpaca_zh-mistral-train.json   \
/mnt/mistral-datasets/alpaca_zh-mistral-valid.json   \
/mnt/mistral-ckpts/Mistral-7B-v0.1-to-mcore-tp4-pp1  \
10000  \
100   \
/workspace/output_mcore_mistral_finetune
```


# 下游任务评估

## Megatron-LM-Dense模型转成Huggingface格式
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/mistral
sh hf2megatron_convertor.sh \
../../../     \
/mnt/mistral-ckpts/Mistral-7B-v0.1-hf-to-megatron-tp4-pp1/release  \
/mnt/mistral-ckpts/Mistral-7B-v0.1-megatron-to-hf    \
4  \
1  \
mistral-7b \
0 \
true
```

## 运行评估工具
```bash
cd /workspace/Pai-Megatron-Patch/LM-Evaluation-Harness-240310
accelerate launch --main_process_port 29051 -m lm_eval \
--model hf \
--model_args pretrained=/mnt/mistral-ckpts/Mistral-7B-v0.1-megatron-to-hf,trust_remote_code=True \
--tasks mmlu,ceval-valid  \
--batch_size 16
```