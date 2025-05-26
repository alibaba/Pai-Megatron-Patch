# Moonlight-16B-A3B模型在Pai-Megatron-Patch的最佳实践

## Table of Contents
   * [安装](#安装)
   * [数据集&模型下载](#数据集和模型下载)
   * [Megatron-Core-MoE模型训练流程](#Megatron-Core-MoE模型训练流程)
      * [模型格式转换](#Megatron-Core-MoE模型格式转换)
      * [继续预训练](#预训练示例)
      * [指令微调](#指令微调示例)
   * [下游任务评估](#下游任务评估)
      * [Megatron-Core-MoE模型格式转换](#评估格式转换)
      * [运行评估工具](#运行评估工具)


## 安装

请在阿里云人工智能平台PAI产品中填写专属镜像地址： `dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/pai-megatron-patch:25.04` 

运行下列代码克隆Pai-Megatron-Patch
```bash
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
cd Pai-Megatron-Patch
```

## 预训练数据集和模型下载

```bash
cd /mnt
mkdir moonlight-ckpts
cd moonlight-ckpts
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/moonlight-ckpts/Moonlight-16B-A3B-Instruct.tar
tar -xvf Moonlight-16B-A3B-Instruct.tar


cd /mnt
mkdir moonlight-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/SlimPajama.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/datasets/alpaca_zh-train-general.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/datasets/alpaca_zh-valid-general.json
```

制作idxmap的脚本如下所示
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/pretrain_data_preprocessing
bash run_make_pretraining_dataset_megatron.sh \
/mnt/moonlight-datasets/SlimPajama.json \
DeepSeekV2Tokenizer \
text \
/mnt/moonlight-datasets/ \
/mnt/moonlight-ckpts/Moonlight-16B-A3B-Instruct
```
为方便期间，我们也提供了已经处理好的idxmap数据集供后续测试使用
```bash
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/moonlight-datasets/mmap_moonlight_datasets_text_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/moonlight-datasets/mmap_moonlight_datasets_text_document.idx
```


## Megatron-Core-MoE模型训练流程
### Megatron-Core-MoE模型格式转换
运行`hf2mcore_moonlight_convertor.sh`脚本，需要传入的参数列表如下
```
MODEL_SIZE=$1                  # 模型参数：A3B
SOURCE_CKPT_PATH=$2            # 源路径
TARGET_CKPT_PATH=$3            # 目标路径
TP=$4                          # 模型并行度
PP=$5                          # 流水并行度
ETP=$6                         # 专家模型并行度
EP=$7                          # 专家并行度
PR=$8                          # 转换精度
mg2hf=$9                       # 是否执行mcore2hf转换
HG_CKPT_PATH=$10               # HF的CKPT的路径
```
例如，使用下述脚本将checkpoint转换到MCore-MoE并检查输出。
注意对于A3B模型由于它有27层，当PP=2时需要执行非均匀切分策略设置`MP_PP0_LAYERS=13`。
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/moonlight

bash hf2mcore_moonlight_convertor.sh \
A3B \
/mnt/moonlight-ckpts/Moonlight-16B-A3B-Instruct \
/mnt/moonlight-ckpts/Moonlight-16B-A3B-Instruct-to-mcore-tp1-pp1-etp1-ep8  \
1 \
1  \
1 \
8 \
bf16 \
false 
```

### Megatron-Core预训练及指令微调
在Moonlight中，我们已将预训练和微调整合到`run_mcore_moonlight.sh`脚本，对于不同的使用场景，二者各参数的意义有所不同。

#### 预训练&微调命令统一描述
需要传入的参数列表如下：
```bash
ENV=$1                          # 运行环境配置开关: dsw单机训练训练，dlc表示多机训练环境
MODEL_SIZE=$2                   # 模型结构参数量级: A37B
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
ETP=${13}                       # 专家张量并行度
EP=${14}                        # 专家并行度
SP=${15}                        # 是否使用序列并行: true, false
DO=${16}                        # 是否使用Megatron版Zero-1降显存优化器: true, false
FL=${17}                        # 是否优先使用Flash Attention: false
SFT=${18}                       # 是否执行微调训练: true, false
AC=${19}                        # 激活检查点模式: sel, full, offload, none
OPTIMIZER_OFFLOAD=${20}         # 是否启用Offload optimizer: false, 或输入0～1的小数作为参数offload比例
SAVE_INTERVAL=${21}             # 保存ckpt的间隔
DATASET_PATH=${22}              # 训练数据集路径
VALID_DATASET_PATH=${23}        # 验证数据集路径
PRETRAIN_CHECKPOINT_PATH=${24}  # 预训练模型路径
TRAIN_TOKENS_OR_ITERS=${25}     # 训练TOKEN或者Iter数
WARMUP_TOKENS_OR_ITERS=${26}    # 预热TOKEN或者Iter数        
OUTPUT_BASEPATH=${27}           # 训练输出日志文件路径
```

#### 预训练示例
使用以下命令启动对Moonlight-16B-A3B的继续预训练。
备注：当`AC=offload`或`full`时，可设置`MP_AC_LAYERS`环境变量来控制Checkpointing或Offload的TransformerLayer层数（默认值：`1`）。

```bash
cd /workspace/Pai-Megatron-Patch/examples/moonlight
bash run_mcore_moonlight.sh  \
dsw  \
A3B   \
1    \
8 \
1e-5   \
1e-6   \
1024  \
1024  \
bf16  \
1   \
1  \
1 \
1 \
8 \
true \
true   \
false \
false \
none   \
false \
100000  \
/mnt/moonlight-datasets/mmap_moonlight_datasets_text_document   \
/mnt/moonlight-datasets/mmap_moonlight_datasets_text_document   \
/mnt/moonlight-ckpts/Moonlight-16B-A3B-Instruct-to-mcore-tp1-pp1-etp1-ep8  \
1000000000  \
10000   \
/workspace/output_mcore_moonlight_pretrain
```

#### 指令微调示例
制作idxmap用于微调的数据集可以参考[链接](https://github.com/alibaba/Pai-Megatron-Patch/tree/main/toolkits/sft_data_preprocessing)。
当准备好微调数据集后，将SFT开关设置为`true`即可进行指令微调。

```bash
cd /workspace/Pai-Megatron-Patch/examples/moonlight
bash run_mcore_moonlight.sh  \
dsw  \
A3B   \
1    \
8 \
1e-5   \
1e-6   \
1024  \
1024  \
bf16  \
1   \
1  \
1 \
1 \
8 \
true \
true   \
false \
true \
none   \
false \
100000  \
/mnt/moonlight-datasets/mmap_moonlight_datasets_text_document   \
/mnt/moonlight-datasets/mmap_moonlight_datasets_text_document   \
/mnt/moonlight-ckpts/Moonlight-16B-A3B-Instruct-to-mcore-tp1-pp1-etp1-ep8  \
10000  \
100   \
/workspace/output_mcore_moonlight_finetune
```
通过设置MP_DATASET_TYPE环境变量，本脚本还可使用json格式的数据集进行指令微调
```bash
export MP_DATASET_TYPE="raw"
cd /workspace/Pai-Megatron-Patch/examples/moonlight
bash run_mcore_moonlight.sh  \
dsw  \
A3B   \
1    \
8 \
1e-5   \
1e-6   \
1024  \
1024  \
bf16  \
1   \
1  \
1 \
1 \
8 \
true \
true   \
false \
true \
none   \
false \
100000  \
/mnt/deepseek-datasets/alpaca_zh-train.json    \
/mnt/deepseek-datasets/alpaca_zh-train.json   \
/mnt/moonlight-ckpts/Moonlight-16B-A3B-Instruct-to-mcore-tp1-pp1-etp1-ep8  \
10000  \
100   \
/workspace/output_mcore_moonlight_finetune
```

## 下游任务评估

### 评估格式转换
您需要将训练/微调后保存的Megatron-Core转换为HuggingFace格式来进行推理评估。

```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/moonlight
bash hf2mcore_moonlight_convertor.sh \
A3B \
/mnt/moonlight-ckpts/Moonlight-16B-A3B-Instruct-to-mcore-tp1-pp1-etp1-ep8  \
/mnt/moonlight-ckpts/Moonlight-16B-A3B-Instruct-mcore-to-hf    \
1  \
1  \
1 \
8 \
bf16 \
true \
/mnt/moonlight-ckpts/Moonlight-16B-A3B-Instruct
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
--model_args pretrained=/mnt/moonlight-ckpts/Moonlight-16B-A3B-Instruct-mcore-to-hf,trust_remote_code=True \
--tasks cmmlu,ceval-valid  \
--batch_size 16
```