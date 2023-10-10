# RLHF
一般来说，SFT微调过的模型在对话场景已经会有不错的表现了。如果想进一步提升模型效果，可以再加上RLHF训练。包括奖励模型（Reward Model）的训练和强化学习（PPO）的训练。这里展示了如何使用当前最常用的RLHF开源代码框架，DeepSpeed-Chat和trlx，来进行奖励函数训练（RM），以及强化学习优化（PPO）。

## 模型格式转换

如果基于huggingface格式的模型直接进行奖励模型训练（RM）和强化学习优化（PPO），可以跳过此步骤。

如果基于Megatron格式的模型，如PAI-Megatron-Patch训练好的SFT模型，进行RM和PPO训练，需要使用我们提供的模型转换脚本，先将Megatron格式的模型文件转换为huggingface格式。

LLaMA2模型转换：
```bash
cd PAI-Megatron-Patch/toolkits/model_checkpoints_convertor/llama2
bash model_convertor.sh \
/path/to/Megatron-LM \
/path/to/megatron_llama2_ckpt \
/path/to/hf_llama2_ckpt \
1 \
1 \
llama-7b \
0 \
true
```
BLOOM模型转换：
```bash
cd PAI-Megatron-Patch/toolkits/model_checkpoints_convertor/bloom
bash model_convertor_huggingface_megatron.sh \
/path/to/Megatron-LM \
/path/to/megatron_bloom_ckpt \
/path/to/hf_bloom_ckpt \
1 \
1 \
true
```

## DeepSpeed-Chat

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

## trlx

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
