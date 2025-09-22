# Qwen3-Next模型在Pai-Megatron-Patch的最佳实践

## Table of Contents
   * [开发环境搭建](#开发环境搭建)
   * [数据集&模型下载](#数据集和模型下载)
   * [Qwen3-Next模型训练流程精简版](#Qwen3-Next模型训练流程精简版)
   * [Qwen3-Next模型训练流程标准版](#Qwen3-Next模型训练流程标准版)
      * [模型格式转换](#模型格式转换)
      * [继续预训练](#预训练示例)
      * [指令微调](#指令微调示例)
   
## 开发环境搭建

请在阿里云Pai平台DSW中基于nvcr.io/nvidia/pytorch:25.06-py3镜像创建开发环境。

```bash
# 修改NGC镜像中的一个小错误
vim /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/driver.py
将第26行修改为以下内容
libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode(errors="ignore")

# 升级Transformer Engine版本
pip uninstall -y transformer-engine transformer-engine-cu12 transformer-engine-torch
git clone --recursive https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine
git submodule update --init --recursive
git checkout c47f329b2084406093124851a3aeecb935183def
python setup.py bdist_wheel  -vvv
cd dist
export NVTE_FRAMEWORK=pytorch 
pip install transformer_engine-2.8.0.dev0+c47f329b-cp312-cp312-linux_x86_64.whl  --no-cache-dir --no-build-isolation -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.cloud.aliyuncs.com

# 安装Transformers
pip install git+https://github.com/huggingface/transformers.git@5f6e278a5177d8b85945a2cdb6b776dacee34914  -i https://mirrors.aliyun.com/pypi/simple/

# 安装Patch依赖
pip install  datasets==3.6.0 packaging==24.2 modelscope -i https://mirrors.aliyun.com/pypi/simple/ 

# 安装triton
pip install --no-build-isolation  "triton==3.2.0" -i https://mirrors.aliyun.com/pypi/simple/

# 安装mamba-ssm
pip install --no-build-isolation  "mamba-ssm" -i https://mirrors.aliyun.com/pypi/simple/

# 安装causal-conv1d
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
git checkout v1.5.2
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
python setup.py bdist_wheel  -vvv
cd dist
export NVTE_FRAMEWORK=pytorch 
pip install causal_conv1d-1.5.2-cp312-cp312-linux_x86_64.whl --no-cache-dir --no-build-isolation -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.cloud.aliyuncs.com

# 安装flash-linear-attention
pip install --no-build-isolation  flash-linear-attention -i https://mirrors.aliyun.com/pypi/simple/

# fix torch.distributed.DistBackendError: NCCL error in: ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1331, unhandled cuda error
pip install --upgrade nvidia-nccl-cu12

```

## 数据集和模型下载

```bash
cd /mnt/data
mkdir qwen-ckpts
cd qwen-ckpts
modelscope download --model Qwen/Qwen3-Next-80B-A3B-Instruct --local_dir Qwen3-Next-80B-A3B-Instruct

cd /mnt/data
mkdir qwen-datasets
cd qwen-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/mmap_qwen3_datasets_text_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/mmap_qwen3_datasets_text_document.idx

wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/datasets/alpaca_zh-train-general.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/datasets/alpaca_zh-valid-general.json

```

## Qwen3-Next模型训练流程精简版
您可以直接将精简版的内容复制到DLC的执行命令栏中进行修改以及训练。精简版将参数分为了三大类：MODEL_ARGS，TRAINING_ARGS以及INFRA_ARGS。
```bash
bash run_mcore_qwen3_lite.sh  \
```

## Qwen3-Next模型训练流程标准版
### 模型格式转换
TBD

### 预训练及指令微调
在Qwen3-Next中，我们已将预训练和微调整合到`run_mcore_qwen3.sh`脚本，对于不同的使用场景，二者各参数的意义有所不同。

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
使用以下命令启动对Qwen3-Next的继续预训练。
备注：当`AC=offload`或`full`时，可设置`MP_AC_LAYERS`环境变量来控制Checkpointing或Offload的TransformerLayer层数（默认值：`1`）。

```bash
cd /workspace/Pai-Megatron-Patch/examples/qwen3_next
bash run_mcore_qwen3.sh  \
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
/mnt/data/qwen-datasets/mmap_qwen3_datasets_text_document  \
/mnt//data/qwen-datasets/mmap_qwen3_datasets_text_document  \
/mnt/data/qwen-ckpts/Qwen3-Next-80B-A3B-Instruct  \
1000000000  \
10000   \
/workspace/output_mcore_qwen3_next_continue_pretrain
```

#### 指令微调示例
制作idxmap用于微调的数据集可以参考[链接](https://github.com/alibaba/Pai-Megatron-Patch/tree/main/toolkits/sft_data_preprocessing)。
当准备好微调数据集后，将SFT开关设置为`true`即可进行指令微调。

```bash
cd /workspace/Pai-Megatron-Patch/examples/qwen3_next
bash run_mcore_qwen3.sh  \
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
/mnt/data/qwen-datasets/mmap_qwen3_datasets_text_document  \
/mnt//data/qwen-datasets/mmap_qwen3_datasets_text_document  \
/mnt/data/qwen-ckpts/Qwen3-Next-80B-A3B-Instruct  \
10000  \
100   \
/workspace/output_mcore_qwen3_next_finetune
```
通过设置MP_DATASET_TYPE环境变量，本脚本还可使用json格式的数据集进行指令微调
```bash
export MP_DATASET_TYPE="raw"
cd /workspace/Pai-Megatron-Patch/examples/qwen3_next
bash run_mcore_qwen3.sh  \
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
/mnt/data/qwen-datasets/alpaca_zh-train-general.json    \
/mnt/data/qwen-datasets/alpaca_zh-valid-general.json   \
/mnt/data/qwen-ckpts/Qwen3-Next-80B-A3B-Instruct  \
10000  \
100   \
/workspace/output_mcore_qwen3_next_finetune
```

