# RLHF
这里展示了如何使用当前最常用的RLHF开源代码框架，DeepSpeed-Chat和trlx，来训练RLHF的奖励函数（RM），以及强化学习优化（PPO）。

## DeepSpeed-Chat

### 安装指南

下载安装开源社区DeepSpeed-Chat源代码：
```bash
git clone https://github.com/microsoft/DeepSpeedExamples.git
cd DeepSpeedExamples/applications/DeepSpeed-Chat/
pip install -r requirements.txt
cp -f rm_main.py ./training/step2_reward_model_finetuning/main.py
```

### 奖励模型训练 (RM)
基于LLaMA2模型训练奖励模型：
```bash
cd training/step2_reward_model_finetuning/ && bash training_scripts/llama2/run_llama2_7b.sh
```

### 强化学习优化（PPO）
基于LLaMA2进行强化学习优化训练：
```bash
cd training/step3_rlhf_finetuning/ && bash training_scripts/llama2/run_llama2_7b_lora.sh
```

## trlx

### 安装指南

下载安装开源社区trlx源代码：
```bash
git clone https://github.com/CarperAI/trlx.git
cp trlx_bloom_rlhf.py trlx_bloom_rlhf_test.py trlx/examples/summarize_rlhf/
cp train_reward_model_bloom.py reward_model_bloom.py ds_config_bloom.json trlx/examples/summarize_rlhf/reward_model/
cp -f ds_config_trlx_gptj_summarize.json trlx/examples/summarize_rlhf/configs/
cd trlx
pip install -e .
```

### 奖励模型训练 (RM)
基于BLOOM模型训练奖励模型：
```bash
cd examples/summarize_rlhf/reward_model/ && deepspeed train_reward_model_bloom.py
```
基于GPT-J模型训练奖励模型：
```bash
cd examples/summarize_rlhf/reward_model/ && deepspeed train_reward_model_gptj.py
```

### 强化学习优化 (PPO)
基于BLOOM模型进行强化学习优化训练：
```bash
cd examples/summarize_rlhf/ && accelerate launch --config_file configs/default_accelerate_config.yaml trlx_bloom_rlhf.py
```
基于GPT-J模型进行强化学习优化训练：
```bash
cd examples/summarize_rlhf/ && accelerate launch --config_file configs/default_accelerate_config.yaml trlx_gptj_text_summarization.py
```

#### PPO单测
如果您想跳过 有监督微调(SFT) 与 奖励模型训练(RM) 两个步骤，只单独测试PPO模块的性能，可以运行如下指令单测PPO：
```bash
cd examples/summarize_rlhf/ && accelerate launch --config_file configs/default_accelerate_config.yaml trlx_bloom_rlhf_test.py
```
