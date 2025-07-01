# Qwen2.5 模型在Pai-Megatron-Patch的最佳实践

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
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-ckpts/Qwen2.5-0.5B.tar.gz
tar -zxf Qwen2.5-0.5B.tar.gz

mkdir qwen-datasets
cd qwen-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/wudao_qwenbpe_text_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/wudao_qwenbpe_text_document.idx

# json微调数据集
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/alpaca_zh-train-general.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/alpaca_zh-valid-general.json
```

## Megatron-Core模型训练流程
### Megatron-Core模型格式转换
当前qwen2.5已升级至`torch_dist`格式权重训练，为了进行权重转换，需要传入的参数列表如下
```
MODEL_SIZE=$1               # 模型大小，0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B
LOAD_DIR=$2                 # 源权重路径
SAVE_DIR=$3                 # 目标权重路径
MG2HF=$4                    # 转换方向 可选: true, false
USE_CUDA=$5                 # 是否使用GPU转换 建议: true
PR=$6                       # 转换精度 可选: fp32 bf16 fp16
HF_DIR=$7                   # HF权重路径(mcore2hf时必须提供)
```
例如，使用下述脚本将checkpoint转换到MCore格式

```bash
cd /workspace/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor
bash scripts/qwen2_5/run_8xH20.sh \
0.5B \
/mnt/qwen-ckpts/Qwen2.5-0.5B-Instruct \
/mnt/qwen-ckpts/Qwen2.5-0.5B-Instruct-to-mcore  \
false \
true \
bf16
```

如果需要自定义转换脚本，请参阅分布式转换工具。

### Megatron-Core预训练及指令微调
在Qwen2.5中，我们已将预训练和微调整合到`run_mcore_qwen.sh`脚本，对于不同的使用场景，二者各参数的意义有所不同。

#### 预训练&微调命令统一描述
需要传入的参数列表如下：
```bash
ENV=$1                          # 运行环境配置开关: dsw单机训练训练，dlc表示多机训练环境
MODEL_SIZE=$2                   # 模型结构参数量级: 0.5B/1.5B/3B/7B/14B/32B/72B
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
SP=${13}                        # 是否使用序列并行: true, false
DO=${14}                        # 是否使用Megatron版Zero-1降显存优化器: true, false
FL=${15}                        # 是否优先使用Flash Attention: true, false
SFT=${16}                       # 是否执行微调训练: true, false
AC=${17}                        # 激活检查点模式: sel, full, offload, false
OPTIMIZER_OFFLOAD=${18}         # 是否启用Offload optimizer: false, static, auto
SAVE_INTERVAL=${19}             # 保存ckpt的间隔
DATASET_PATH=${20}              # 训练数据集路径
VALID_DATASET_PATH=${21}        # 验证数据集路径
PRETRAIN_CHECKPOINT_PATH=${22}  # 预训练模型路径
TRAIN_TOKENS_OR_ITERS=${23}     # 训练TOKEN或者Iter数
WARMUP_TOKENS_OR_ITERS=${24}    # 预热TOKEN或者Iter数        
OUTPUT_BASEPATH=${25}           # 训练输出日志文件路径
```

#### 预训练示例
使用以下命令启动对qwen2的继续预训练。
备注：当`AC=offload`或`full`时，可设置`MP_AC_LAYERS`环境变量来控制Checkpointing或Offload的TransformerLayer层数（默认值：`1`）。

```bash
cd /workspace/Pai-Megatron-Patch/examples/qwen2_5
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
true \
true   \
true \
false \
false   \
false \
100000  \
/mnt/qwen-datasets/wudao_qwenbpe_text_document   \
/mnt/qwen-datasets/wudao_qwenbpe_text_document   \
/mnt/qwen-ckpts/Qwen2.5-0.5B-Instruct-to-mcore  \
10000  \
100   \
/workspace/output_mcore_qwen2.5_pretrain
```

#### 指令微调示例
制作idxmap用于微调的数据集可以参考[链接](https://github.com/alibaba/Pai-Megatron-Patch/tree/main/toolkits/sft_data_preprocessing)。
当准备好微调数据集后，将SFT开关设置为`true`即可进行指令微调。

```bash
cd /workspace/Pai-Megatron-Patch/examples/qwen2_5
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
/workspace/output_mcore_qwen2.5_finetune
```
通过设置MP_DATASET_TYPE环境变量，本脚本还可使用json格式的数据集进行指令微调
```bash
export MP_DATASET_TYPE="raw"
cd /workspace/Pai-Megatron-Patch/examples/qwen2_5
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
true \
true   \
true \
true \
false   \
false \
100000  \
/mnt/qwen-datasets/alpaca_zh-qwen-train.json    \
/mnt/qwen-datasets/alpaca_zh-qwen-valid.json   \
/mnt/qwen-ckpts/Qwen2.5-0.5B-Instruct-to-mcore  \
10000  \
100   \
/workspace/output_mcore_qwen2.5_finetune
```

## 下游任务评估

### 评估格式转换
您需要将训练/微调后保存的Megatron-Core转换为HuggingFace格式来进行推理评估。

```bash
cd /workspace/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor
bash hf2mcore_qwen2.5_convertor.sh \
0.5B \
/mnt/qwen-ckpts/Qwen2.5-0.5B-Instruct-to-mcore  \
/mnt/qwen-ckpts/Qwen2.5-0.5B-Instruct-back    \
1  \
1  \
bf16 \
true \
true \
/mnt/qwen-ckpts/Qwen2.5-0.5B-Instruct
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
--model_args pretrained=/mnt/qwen-ckpts/Qwen2.5-0.5B-Instruct-back,trust_remote_code=True \
--tasks cmmlu,ceval-valid  \
--batch_size 16
```