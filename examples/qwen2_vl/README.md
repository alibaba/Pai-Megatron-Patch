# Qwen2-VL模型在Pai-Megatron-Patch的最佳实践

## Table of Contents
   * [安装](#安装)
   * [数据集&模型下载](#数据集和模型下载)
   * [Megatron-Core模型训练流程](#Megatron-Core模型训练流程)
      * [模型格式转换](#Megatron-Core模型格式转换)
      * [继续预训练](#预训练示例)

## 安装

请在阿里云人工智能平台PAI产品中填写专属镜像地址： `dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/pai-megatron-patch-vlm:24.11` 

运行下列代码克隆Pai-Megatron-Patch
```bash
cd /workspace
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
```

## 数据集和模型下载

```bash
cd /mnt
mkdir qwen2-vl-ckpts
cd qwen2-vl-ckpts
git clone https://www.modelscope.cn/Qwen/Qwen2-VL-7B-Instruct.git
cd ..

mkdir llava-datasets
cd llava-datasets
git clone https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain
cd LLaVA-Pretrain
unzip images.zip

#convert to webdataset format:
cd /workspace/Pai-Megatron-Patch/toolkits/pretrain_data_preprocessing
python convert_llava_pretrain_to_wds.py /mnt/llava-datasets/LLaVA-Pretrain/

#convert to megatron-energon format:
cd /mnt/llava-datasets/LLaVA-Pretrain/wds
energon prepare ./

#select the following values for the presented options:
> Please enter a desired train/val/test split like "0.5, 0.2, 0.3" or "8,1,1": 9,1,0
> Do you want to create a dataset.yaml interactively? [Y/n]: Y
> Please enter a number to choose a class: 10 (VQAWebdataset)
> Do you want to set a simple field_map[Y] (or write your own sample_loader [n])? [Y/n]: Y
> Please enter a webdataset field name for 'image' (<class 'torch.Tensor'>): jpg
> Please enter a webdataset field name for 'context' (<class 'str'>): json[0][value]
> Please enter a webdataset field name for 'answers' (typing.Optional[typing.List[str]], default: None): json[1][value]
> Please enter a webdataset field name for 'answer_weights' (typing.Optional[torch.Tensor], default: None):
```
为方便起见，我们也提供了处理好的wds文件(26G)用于后续的测试，下载链接如下所示:
```bash
cd /mnt/llava-datasets/LLaVA-Pretrain/
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/vlm-datasets/wds.tgz
tar -zxf wds.tgz
```

## Megatron-Core模型训练流程
### Megatron-Core模型格式转换
运行`hf2mcore_qwen2_vl_convertor.sh`脚本，需要传入的参数列表如下
```bash
MODEL_SIZE=$1                 # 模型参数：2B/7B/72B
SOURCE_CKPT_PATH=$2           # 源llm checkpoint路径
TARGET_CKPT_PATH=$3           # 目标checkpoint路径
TP=$4                         # 解码器模型并行度(目前仅支持1)
PP=$5                         # 解码器流水并行度(目前仅支持1)
mg2hf=$6                      # 是否执行mcore2hf转换
PR=$7                         # 精度设置，fp16/bf16/fp32     
HF_CKPT_PATH=$8               # HF的CKPT的路径【可选，mg2hf=true时必须提供】
```
例如，使用下述脚本将checkpoint转换到MCore-Dense并检查输出

```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen
bash hf2mcore_qwen2_vl_convertor.sh \
7B \
/mnt/qwen2-vl-ckpts/Qwen2-VL-7B-Instruct \
/mnt/qwen2-vl-ckpts/Qwen2-VL-7B-Instruct-tp1pp1 \
1  \
1  \
false \
bf16
```

### Megatron-Core预训练

#### 预训练命令描述
需要传入的参数列表如下：
```bash
ENV=$1                          # 运行环境配置开关: dsw单机训练训练，dlc表示多机训练环境
MODEL_SIZE=$2                   # 模型结构参数量级: 2B/7B/72B
BATCH_SIZE=$3                   # 一次迭代一个数据并行内的样本数
GLOBAL_BATCH_SIZE=$4            # 一次迭代多个数据并行的总样本数
LR=$5                           # 学习率
MIN_LR=$6                       # 最小学习率
SEQ_LEN=$7                      # 序列长度
PAD_LEN=$8                      # Padding后长度
PR=${9}                         # 训练精度: fp16, bf16, fp8
TP=${10}                        # 模型并行度
PP=${11}                        # 流水并行度
CP=${12}                        # 上下文并行度
DO=${13}                        # 是否使用Megatron版Zero-1降显存优化器: true, false
FL=${14}                        # 是否优先使用Flash Attention: true, false
AC=${15}                        # 激活检查点模式: sel, full, offload, false
OPTIMIZER_OFFLOAD=${16}         # 是否启用Offload optimizer: false, static, auto
SAVE_INTERVAL=${17}             # 保存ckpt的间隔
DATASET_PATH=${18}              # 训练数据集路径
VALID_DATASET_PATH=${19}        # 验证数据集路径
PRETRAIN_CHECKPOINT_PATH=${20}  # 预训练模型路径
TRAIN_ITERS=${21}               # Iter数
LR_WARMUP_ITERS=${22}           # 预热Iter数        
OUTPUT_BASEPATH=${23}           # 训练输出日志文件路径
```

#### 预训练示例
使用以下命令启动对Qwen2-VL的继续预训练。

```bash
cd /workspace/Pai-Megatron-Patch/examples/qwen2_vl
sh run_mcore_qwen.sh  \
dsw  \
7B   \
1    \
256 \
0.00015   \
1e-5   \
1024  \
1024  \
bf16  \
1   \
1  \
1 \
true \
true   \
true \
false \
100000  \
/mnt/llava-datasets/LLaVA-Pretrain/wds   \
/mnt/llava-datasets/LLaVA-Pretrain/wds   \
/mnt/qwen2-vl-ckpts/Qwen2-VL-7B-Instruct-tp1pp1 \
20000  \
200   \
/workspace/output_mcore_qwen2vl_pretrain
```