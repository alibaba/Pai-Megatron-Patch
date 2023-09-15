## RLHF
这里展示了如何使用DeepSpeed-Chat代码库，进行奖励函数训练（RM），以及强化学习优化（PPO）。如果基于Megatron格式的SFT模型，需要先将Megatron格式的模型文件转换为huggingface格式，具体可以参考[这里](../README.md)。

### 安装指南

下载安装开源社区DeepSpeed-Chat源代码：
```bash
cd PAI-Megatron-Patch/rlhf/deepspeed-chat
git clone https://github.com/microsoft/DeepSpeedExamples.git
cp -f rm_main.py DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/main.py
cp -f utils.py DeepSpeedExamples/applications/DeepSpeed-Chat/training/utils/utils.py
cd DeepSpeedExamples/applications/DeepSpeed-Chat/
pip install -r requirements.txt
```

### 奖励模型训练（RM）
基于LLaMA2模型训练奖励模型：
```bash
cd training/step2_reward_model_finetuning/ && bash training_scripts/llama2/run_llama2_7b.sh
```

### 强化学习优化（PPO）
基于LLaMA2进行强化学习优化训练：
```bash
cd training/step3_rlhf_finetuning/ && bash training_scripts/llama2/run_llama2_7b_lora.sh
```
