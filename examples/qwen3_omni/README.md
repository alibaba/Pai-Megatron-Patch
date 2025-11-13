# Qwen3-Omni模型在Pai-Megatron-Patch的最佳实践

## Table of Contents
   * [开发&运行环境搭建](#开发&运行环境搭建)
   * [微调模型&数据集准备](#微调模型&数据集准备)
   * [Megatron-Core模型训练流程](#Megatron-Core模型训练流程)
      * [模型格式转换](#模型格式转换)
      * [训练代码调试](#训练代码调试)
      * [模型微调](#Qwen3-Omni-30B-A3B-Instruct微调示例)

## 开发&运行环境搭建

请在阿里云Pai平台DSW中基于nvcr.io/nvidia/pytorch:24.12-py3镜像创建开发环境。

```bash
# 安装支持qwen3-omni的transformers
pip install git+https://github.com/huggingface/transformers.git@4d0b6758b90aa54e4077171e6d42c55e0c01c622  -i http://mirrors.cloud.aliyuncs.com/pypi/simple --trusted-host mirrors.cloud.aliyuncs.com

# 安装torchaudio，这会导致重新安装pytorch==2.8.0
pip install torchaudio==2.8.0 torchvision==0.23.0 -i http://mirrors.cloud.aliyuncs.com/pypi/simple --trusted-host mirrors.cloud.aliyuncs.com

# 重装pytorch2.8后需要重新安装flash_attn以及apex
pip uninstall -y flash_attn && pip install https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/csrc/flash-attention/torch2.8.0-cu12x/flash_attn-2.7.4.post1-cp312-cp312-linux_x86_64.whl
pip uninstall -y apex && pip install https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/csrc/apex/torch2.8.0-cuda12x/apex-0.1-cp312-cp312-linux_x86_64.whl --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ 

# 升级Transformer Engine到2.7
pip uninstall -y transformer-engine transformer-engine-cu12 transformer-engine-torch
git clone --recursive https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine
git submodule update --init --recursive
git checkout release_v2.7
export CUDNN_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/
rm -rf /usr/local/lib/python3.12/dist-packages/nvidia/cudnn/include/__pycache__
cp /usr/local/lib/python3.12/dist-packages/nvidia/cudnn/include/*  /usr/local/cuda/include/
python setup.py bdist_wheel  -vvv
cd dist
export NVTE_FRAMEWORK=pytorch 
pip install onnxscript==0.3.1 -i https://mirrors.aliyun.com/pypi/simple/
pip install transformer_engine-2.7.0-cp312-cp312-linux_x86_64.whl --no-cache-dir -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.cloud.aliyuncs.com

# 安装Patch依赖
pip install megatron-energon==4.0.0 webdataset==0.2.110 datasets==3.6.0 packaging==24.2 modelscope==1.30.0  -i http://mirrors.cloud.aliyuncs.com/pypi/simple --trusted-host mirrors.cloud.aliyuncs.com

# 升级multi-storage-client
pip install -U multi-storage-client  -i http://mirrors.cloud.aliyuncs.com/pypi/simple --trusted-host mirrors.cloud.aliyuncs.com


# 下载Pai-Megatron-Patch源码
cd /mnt
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git

```

## 微调模型&数据集准备

使用modelscope来快速下载Qwen3-Omni-30B-A3B-Instruct模型
```bash
cd /mnt
mkdir qwen3-omni-ckpts
cd qwen3-omni-ckpts
modelscope download --model Qwen/Qwen3-Omni-30B-A3B-Instruct --local_dir Qwen3-Omni-30B-A3B-Instruct
```
我们提供了基于DataJuicer的大规模且高性能的多模态数据序列化的方案，在实施该方案前我们通过一个简单的demo熟悉下多模态数据序列化流程。 
build_fake_wds_for_omni.py脚本是将原始的多模态多轮对话数据通过[webdataset](https://github.com/webdataset/webdataset)打包成[Energon](https://github.com/NVIDIA/Megatron-Energon)数据加载器可识别的格式。
```bash
cd /mnt/Pai-Megatron-Patch/toolkits/multimodal_data_preprocessing
python build_fake_wds_for_omni.py --output-dir wds
```
当熟悉了多模态数据序列化逻辑后，我们就可以参考[DataJuicer](./README_DataJuicer.md)指引来对多模态数据进行大规模处理了。
使用DJ生成TAR包后，需要调用Patch的脚本生成元数据，如下所示:
```bash
cd /mnt/Pai-Megatron-Patch/toolkits/multimodal_data_preprocessing
python build_wds_meta_data_from_datajuice.py --dataset-root ${your dj output dir}
```

## Megatron-Core模型训练流程
### 模型格式转换
```bash
cd /mnt/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor
bash scripts/qwen3_omni/run_8xH20.sh \
A3B \
/mnt/qwen3-omni-ckpts/Qwen3-Omni-30B-A3B-Instruct  \
/mnt/qwen3-omni-ckpts/Qwen3-Omni-30B-A3B-Instruct-to-mcore  \
false \
true \
bf16
```

### 训练代码调试
由于Qwen3-omni的多模态数据融合以及模型结构复杂，我们提供了用于单机调试的脚本方便理解代码逻辑或者快速发现Issue等等。
启动单机训练代码调试的脚本如下：
```bash
cd /mnt/Pai-Megatron-Patch/examples/qwen3_omni
bash debug_mcore_qwen3_omni.sh
```

### Qwen3-Omni-30B-A3B-Instruct微调示例
#### 微调示例
使用以下命令启动对Qwen3-Omni-30B-A3B-Instruct的微调。
```bash
cd /mnt/Pai-Megatron-Patch/examples/qwen3_omni
bash run_mcore_qwen3_omni.sh
```