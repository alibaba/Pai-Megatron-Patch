## RLHF
这里展示了如何使用trlx代码库，进行奖励函数训练（RM），以及强化学习优化（PPO）。如果基于Megatron格式的SFT模型，需要先将Megatron格式的模型文件转换为huggingface格式，具体可以参考[这里](../README.md)。

### 安装指南

下载安装开源社区trlx源代码：
```bash
cd PAI-Megatron-Patch/rlhf/trlx
git clone https://github.com/CarperAI/trlx.git
cp trlx_bloom_rlhf.py trlx_bloom_rlhf_test.py trlx/examples/summarize_rlhf/
cp train_reward_model_bloom.py reward_model_bloom.py ds_config_bloom.json trlx/examples/summarize_rlhf/reward_model/
cp -f ds_config_trlx_gptj_summarize.json trlx/examples/summarize_rlhf/configs/
cd trlx
pip install -e .
```

### 奖励模型训练（RM）
基于BLOOM模型训练奖励模型：
```bash
cd examples/summarize_rlhf/reward_model/ && deepspeed train_reward_model_bloom.py
```
基于GPT-J模型训练奖励模型：
```bash
cd examples/summarize_rlhf/reward_model/ && deepspeed train_reward_model_gptj.py
```

### 强化学习优化（PPO）
基于BLOOM模型进行强化学习优化训练：
```bash
cd examples/summarize_rlhf/ && accelerate launch --config_file configs/default_accelerate_config.yaml trlx_bloom_rlhf.py
```
基于GPT-J模型进行强化学习优化训练：
```bash
cd examples/summarize_rlhf/ && accelerate launch --config_file configs/default_accelerate_config.yaml trlx_gptj_text_summarization.py
```

#### PPO单测
如果想跳过 有监督微调(SFT) 与 奖励模型训练(RM) 两个步骤，只单独测试PPO模块的性能，可以运行如下指令单测PPO：
```bash
cd examples/summarize_rlhf/ && accelerate launch --config_file configs/default_accelerate_config.yaml trlx_bloom_rlhf_test.py
```