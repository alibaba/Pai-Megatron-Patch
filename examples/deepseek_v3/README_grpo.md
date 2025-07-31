# 基于 Mcore 的端到端GRPO训练流程

本文档提供使用 ChatLearn、Mcore 和 vLLM 框架来对DeepSeek-V3模型进行GRPO训练的快速开始指南。

## 环境配置
1. Docker镜像准备
我们建议在PAI [DSW](https://help.aliyun.com/zh/pai/user-guide/create-and-manage-dsw-instances/)/[DLC](https://help.aliyun.com/zh/pai/user-guide/create-a-training-task?spm=a2c4g.11186623.help-menu-30347.d_3_3_5_5.2dfb1925l3QjwG)中运行该示例，你需要填写如下镜像地址来启动实例：
```bash
dsw-registry.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.6.0-vllm0.8.5-ubuntu24.04-cuda12.6-py312
```

可以使用vpc地址来加速镜像拉取速度，需要根据当前region信息来更改镜像地址。比如，启动在上海的DSW实例，可以使用如下镜像`dsw-registry-vpc.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.6.0-vllm0.8.5-ubuntu24.04-cuda12.6-py312
`。

2. 代码准备

```bash
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
```

## 数据准备
以[MATH-lighteval](https://www.modelscope.cn/datasets/AI-ModelScope/MATH-lighteval)数据集作为示例.
```bash
# 下载数据集
mkdir -p /mnt/data/datasets
modelscope download --dataset AI-ModelScope/MATH-lighteval --local_dir /mnt/data/datasets/MATH-lighteval
# 数据集预处理
python examples/fsdp/data/data_preprocess/math_lighteval.py --input_dir /mnt/data/datasets/MATH-lighteval --local_dir /mnt/data/datasets/MATH-lighteval
# 下载模型权重
modelscope download --model deepseek-ai/DeepSeek-V3-0324 --local_dir /mnt/data/ckpts/huggingface/DeepSeek-V3-0324
```

## 模型转换

模型格式转换可以参考 [Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch) 项目提供的转换脚本。
高性能分布式权重转换可以参考：https://github.com/alibaba/Pai-Megatron-Patch/tree/main/toolkits/distributed_checkpoints_convertor


例如，使用下述脚本将671B量级的DeepSeek-V3的Huggingface格式的模型转换到MCore格式
```bash
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
cd ~/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/deepseek
python fp8_cast_bf16.py --input-fp8-hf-path /mnt/data/ckpts/huggingface/DeepSeek-V3-0324 --output-bf16-hf-path /mnt/data/ckpts/huggingface/DeepSeek-V3-0324-BF16

cd ~/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor
bash scripts/deepseek_v3/run_32xH20.sh \
A37B  \
/mnt/data/ckpts/huggingface/DeepSeek-V3-0324-BF16 \
/mnt/data/ckpts/mcore/DeepSeek-V3-0324-BF16-to-mcore  \
false \
true \
bf16
```

## 训练
运行以下命令开始训练：

```bash
cd ~/Pai-Megatron-Patch/examples/deepseek_v3
bash run_mcore_deepseek_grpo.sh
```

## 使用 Wandb 监控
如需使用 Wandb 记录训练过程，请参考如下配置：

```bash
export enable_wandb=True
export wandb_project="Your-Wandb-Project-Name"
export WANDB_API_KEY="Your-Wandb-api-key"
```