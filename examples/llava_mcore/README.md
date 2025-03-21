# LLaVA模型在Pai-Megatron-Patch的最佳实践

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
mkdir mistral-clip-ckpts
cd mistral-clip-ckpts
git clone https://modelscope.cn/models/rubraAI/Mistral-7B-Instruct-v0.3
git clone https://modelscope.cn/models/AI-ModelScope/clip-vit-large-patch14-336

mkdir llava-datasets
cd llava-datasets
git clone https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain
cd LLaVA-Pretrain
unzip images.zip

#convert to webdataset format:
cd /workspace/Pai-Megatron-Patch/toolkits/multimodal_data_preprocessing
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
运行`hf2mcore_convertor_llava.sh`脚本，需要传入的参数列表如下
```bash
MODEL_SIZE=$1                 # 模型参数：7B
SOURCE_LLM_CKPT_PATH=$2       # 源llm checkpoint路径
SOURCE_CLIP_CKPT_PATH=$3      # 源clip checkpoint路径
TARGET_CKPT_PATH=$4           # 目标checkpoint路径
TP=$5                         # 模型并行度
PP=$6                         # 流水并行度
mg2hf=$7                      # 是否执行mcore2hf转换
PR=$8                      # 精度设置，fp16/bf16/fp32     
HF_CKPT_PATH=$9            # HF的CKPT的路径【可选，mg2hf=true时必须提供】
```
例如，使用下述脚本将checkpoint转换到MCore-Dense并检查输出

```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/llava
bash hf2mcore_convertor_llava.sh \
7B \
/mnt/mistral-clip-ckpts/Mistral-7B-Instruct-v0.3 \
/mnt/mistral-clip-ckpts/clip-vit-large-patch14-336  \
/mnt/mistral-clip-ckpts/Mistral-7B-Instruct-v0.3-to-mcore-tp4-pp1 \
4  \
1  \
false \
bf16
```

### Megatron-Core预训练

#### 预训练命令描述
需要传入的参数列表如下：
```bash
ENV=$1                          # 运行环境配置开关: dsw单机训练训练，dlc表示多机训练环境
MODEL_SIZE=$2                   # 模型结构参数量级: 0.5B/1.5B/3B/7B/14B/32B/72B
BATCH_SIZE=$3                   # 一次迭代一个数据并行内的样本数
GLOBAL_BATCH_SIZE=$4            # 一次迭代多个数据并行的总样本数
LR=$5                           # 学习率
MIN_LR=$6                       # 最小学习率
SEQ_LEN=$7                      # 序列长度
DECODER_SEQ_LEN=$8              # 解码序列长度
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
TRAIN_ITERS=${21}               # 训练TOKEN或者Iter数
LR_WARMUP_ITERS=${22}           # 预热TOKEN或者Iter数        
OUTPUT_BASEPATH=${23}           # 训练输出日志文件路径
```

#### 预训练示例
使用以下命令启动对llaVA的继续预训练。
备注：当加载mcore模型时出现无法加载`extra_states`时，可设置`Megatron-LM-241113/megatron/training/checkpointing`的1168行的`strict`为`False`。

```bash
cd /workspace/Pai-Megatron-Patch/examples/llava_mcore
sh run_mcore_llava.sh  \
dsw  \
7B   \
1    \
256 \
0.00015   \
1e-5   \
576  \
1024  \
bf16  \
4   \
1  \
1 \
true \
true   \
true \
false \
100000  \
/mnt/llava-datasets/LLaVA-Pretrain/wds   \
/mnt/llava-datasets/LLaVA-Pretrain/wds   \
/mnt/mistral-clip-ckpts/Mistral-7B-Instruct-v0.3-to-mcore-tp4-pp1 \
20000  \
200   \
/workspace/output_mcore_llava_pretrain
```

使用上述命令训练20k iters后，您应当能在tensorboard内看到如下loss曲线：
<div align=center>
<img src=lm_loss.png width=1200 height=400 />
</div>
