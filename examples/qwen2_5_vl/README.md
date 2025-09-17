# Qwen2.5-VL模型在Pai-Megatron-Patch的最佳实践

## Table of Contents
   * [安装](#安装)
   * [数据集&模型下载](#数据集和模型下载)
   * [Megatron-Core模型训练流程](#Megatron-Core模型训练流程)
      * [模型格式转换](#Megatron-Core模型格式转换)
      * [继续预训练](#预训练示例)

## 安装

请在阿里云人工智能平台PAI产品中填写专属镜像地址： `dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/pai-megatron-patch:25.01` 

运行下列代码克隆Pai-Megatron-Patch
```bash
cd /workspace
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
```

## 数据集和模型下载

```bash
cd /mnt
mkdir qwen2.5-vl-ckpts
cd qwen2.5-vl-ckpts
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-ckpts/Qwen2.5-VL-3B-Instruct.tar
tar -xvf Qwen2.5-VL-3B-Instruct.tar
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

对于视频多模态、单样本中包含多张图片、多轮对话等复杂数据集，您需要将其转换为sharegpt格式数据后再使用Megatron-Patch训练。对于sharegpt格式的数据处理，参见[链接](../../toolkits/multimodal_data_preprocessing/dataset_preparation.md)。


## Megatron-Core模型训练流程
### Megatron-Core模型格式转换
当前qwen2.5-VL已升级至`torch_dist`格式权重训练，为了进行权重转换，需要传入的参数列表如下
```
MODEL_SIZE=$1               # 模型大小，3B, 7B, 32B, 72B
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
bash scripts/qwen2_5_vl/run_8xH20.sh \
3B \
/mnt/qwen2.5-vl-ckpts/Qwen2.5-VL-3B-Instruct \
/mnt/qwen2.5-vl-ckpts/Qwen2.5-VL-3B-Instruct-to-mcore  \
false \
true \
bf16
```

当您需要将训练好的checkpoint转换回huggingface格式用于推理时，执行

```bash
cd /workspace/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor
bash scripts/qwen2_5_vl/run_8xH20.sh \
3B \
/mnt/qwen2.5-vl-ckpts/Qwen2.5-VL-3B-Instruct-to-mcore \
/mnt/qwen2.5-vl-ckpts/Qwen2.5-VL-3B-Instruct-back  \
true \
true \
bf16 \
/mnt/qwen2.5-vl-ckpts/Qwen2.5-VL-3B-Instruct-back
```

### Megatron-Core预训练

> 关于attention: Qwen2.5-VL调用了varlen attention，若您使用Hopper架构GPU，推荐将FL设为false以使用FusedAttention后端来获得最佳性能；
对于其他NVIDIA GPU，由于FusedAttention不支持varlen，请将FL设置为true。此外，目前观察到Flash-Attention 3会出现不正常的grad norm，不推荐使用。

#### 预训练命令描述
需要传入的参数列表如下：
```bash
ENV=$1                          # 运行环境配置开关: dsw单机训练训练，dlc表示多机训练环境
MODEL_SIZE=$2                   # 模型结构参数量级: 3B/7B/72B
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
SP=${13}                        # 是否启用序列并行: true, false
DO=${14}                        # 是否使用Megatron版Zero-1降显存优化器: true, false
FL=${15}                        # 是否优先使用Flash Attention: true, false
AC=${16}                        # 激活检查点模式: sel, full, offload, false
OPTIMIZER_OFFLOAD=${17}         # 是否启用Offload optimizer: false, 或输入0～1的小数作为参数offload比例
SAVE_INTERVAL=${18}             # 保存ckpt的间隔
DATASET_PATH=${19}              # 训练数据集路径
VALID_DATASET_PATH=${20}        # 验证数据集路径
PRETRAIN_CHECKPOINT_PATH=${21}  # 预训练模型路径
TRAIN_ITERS=${22}               # Iter数
LR_WARMUP_ITERS=${23}           # 预热Iter数        
OUTPUT_BASEPATH=${24}           # 训练输出日志文件路径
```

#### 预训练示例
使用以下命令启动对Qwen2.5-VL的继续预训练。

```bash
cd /workspace/Pai-Megatron-Patch/examples/qwen2_5_vl
bash run_mcore_qwen.sh  \
dsw  \
3B   \
1    \
32 \
1e-5   \
1e-6   \
2048  \
2048  \
bf16  \
2   \
2  \
1 \
true \
true \
true   \
false \
false \
100000  \
/mnt/llava-datasets/LLaVA-Pretrain/wds   \
/mnt/llava-datasets/LLaVA-Pretrain/wds   \
/mnt/qwen2.5-vl-ckpts/Qwen2.5-VL-3B-Instruct-to-mcore \
20000  \
200   \
/workspace/output_mcore_qwen2_5_vl_pretrain
```

由于PP切分时，PP Rank 0额外的ViT会导致其负载略高于其他PP Rank，为了达到最佳性能，您可能需要调整`MP_PP0_LAYERS`变量降低PP Rank 0的LLM层数。

```bash
cd /workspace/Pai-Megatron-Patch/examples/qwen2_5_vl
MP_PP0_LAYERS=16 sh run_mcore_qwen.sh  \
dsw  \
3B   \
1    \
32 \
1e-5   \
1e-6   \
2048  \
2048  \
bf16  \
2   \
2  \
1 \
true \
true \
true   \
false \
false \
100000  \
/mnt/llava-datasets/LLaVA-Pretrain/wds   \
/mnt/llava-datasets/LLaVA-Pretrain/wds   \
/mnt/qwen2.5-vl-ckpts/Qwen2.5-VL-3B-Instruct-to-mcore \
20000  \
200   \
/workspace/output_mcore_qwen2_5_vl_pretrain
```