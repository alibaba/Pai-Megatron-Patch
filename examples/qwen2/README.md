# Qwen2 模型在Pai-Megatron-Patch的最佳实践

## Table of Contents
   * [安装](#安装)
   * [数据集&模型下载](#数据集和模型下载)
   * [Megatron-Core模型训练流程](#Megatron-Core模型训练流程)
      * [模型格式转换](#Megatron-Core模型格式转换)
      * [继续预训练](#预训练示例)
      * [指令微调](#指令微调示例)
   * [下游任务评估](#下游任务评估)
      * [Megatron-Core模型格式转换](#评估格式转换)
      * [运行评估工具](#运行评估工具)


## 安装

请在阿里云人工智能平台PAI产品中填写专属镜像地址： `dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/pai-megatron-patch:25.01` 

运行下列代码克隆Pai-Megatron-Patch
```bash
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
cd Pai-Megatron-Patch
```

## 预训练数据集和模型下载

```bash
cd /mnt
mkdir qwen-ckpts
cd qwen-ckpts
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-ckpts/Qwen2-0.5B.tgz
tar -zxf Qwen2-0.5B.tgz

mkdir qwen-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/wudao_qwenbpe_text_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/wudao_qwenbpe_text_document.idx

wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/alpaca_zh-train-general.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/alpaca_zh-valid-general.json


```

## Megatron-Core模型训练流程
### Megatron-Core模型格式转换
运行`hf2mcore_qwen2_convertor.sh`脚本，需要传入的参数列表如下
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
HG_CKPT_PATH=${10}             # HF的CKPT的路径
```
例如，使用下述脚本将checkpoint转换到MCore-Dense并检查输出

```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen
bash hf2mcore_qwen2_convertor.sh \
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
还使用下述脚本将checkpoint转换到MCore-MoE并检查输出
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

### Megatron-Core预训练及指令微调
在Qwen2中，我们已将预训练和微调整合到`run_mcore_qwen.sh`脚本，对于不同的使用场景，二者各参数的意义有所不同。

#### 预训练&微调命令统一描述
需要传入的参数列表如下：
```bash
ENV=$1                          # 运行环境配置开关: dsw单机训练训练，dlc表示多机训练环境
MODEL_SIZE=$2                   # 模型结构参数量级: 7B, 72B, A14B
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
使用以下命令启动对qwen2的继续预训练。
备注：当`AC=offload`或`full`时，可设置`MP_AC_LAYERS`环境变量来控制Checkpointing或Offload的TransformerLayer层数（默认值：`1`）。

```bash
cd /workspace/Pai-Megatron-Patch/examples/qwen2
sh run_mcore_qwen.sh  \
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
1 \
true \
true   \
true \
false \
false   \
false \
100000  \
/mnt/qwen-datasets/wudao_qwenbpe_text_document   \
/mnt/qwen-datasets/wudao_qwenbpe_text_document   \
/mnt/qwen-ckpts/Qwen2-0.5B-hf-to-mcore-te-tp1-pp1  \
10000  \
100   \
/workspace/output_mcore_qwen2_pretrain
```

#### 指令微调示例
制作idxmap用于微调的数据集可以参考[链接](https://github.com/alibaba/Pai-Megatron-Patch/tree/main/toolkits/sft_data_preprocessing)。
当准备好微调数据集后，将SFT开关设置为`true`即可进行指令微调。

```bash
cd /workspace/Pai-Megatron-Patch/examples/qwen2
sh run_mcore_qwen.sh  \
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
/workspace/output_mcore_qwen2_finetune
```
通过设置MP_DATASET_TYPE环境变量，本脚本还可使用json格式的数据集进行指令微调
```bash
export MP_DATASET_TYPE="raw"
cd /workspace/Pai-Megatron-Patch/examples/qwen2
sh run_mcore_qwen.sh  \
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
1 \
true \
true   \
true \
true \
false   \
false \
100000  \
/mnt/qwen-datasets/alpaca_zh-qwen-train.json    \
/mnt/qwen-datasets/alpaca_zh-qwen-valid.json   \
/mnt/qwen-ckpts/Qwen2-0.5B-hf-to-mcore-te-tp1-pp1  \
10000  \
100   \
/workspace/output_mcore_qwen2_finetune
```

## 下游任务评估

### 评估格式转换
您需要将训练/微调后保存的Megatron-Core转换为HuggingFace格式来进行推理评估。

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

### 运行评估工具
下载评估数据
```bash
# In container
cd /workspace

wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/evaluation-datasets/evaluate.tgz 
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/evaluation-datasets/cmmlu.tgz 
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/evaluation-datasets/ceval.tgz 

tar -xvzf cmmlu.tgz 
tar -xvzf ceval.tgz 
tar -xvzf evaluate.tgz
```
运行以下指令对转换后的模型进行评估。
```bash
cd /workspace/Pai-Megatron-Patch/LM-Evaluation-Harness-240310
accelerate launch --main_process_port 29051 -m lm_eval \
--model hf \
--model_args pretrained=/mnt/qwen-ckpts/Qwen2-0.5B-mcore-te-to-hf,trust_remote_code=True \
--tasks cmmlu,ceval-valid  \
--batch_size 16
```