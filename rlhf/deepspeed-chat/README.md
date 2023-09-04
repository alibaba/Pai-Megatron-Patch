## RLHF
这里展示了如何使用DeepSpeed-Chat库训练RLHF的奖励函数（RM）。

### 安装指南

下载安装开源社区DeepSpeed-Chat源代码：
```bash
git clone https://github.com/microsoft/DeepSpeedExamples.git
cd DeepSpeedExamples/applications/DeepSpeed-Chat/
pip install -r requirements.txt
cp -f deepspeed_bloom_main.py ./training/step2_reward_model_finetuning/main.py
```

### 奖励模型训练 (RM)
训练奖励模型：
```bash
cd training/step2_reward_model_finetuning/ && bash training_scripts/single_node/run_350m.sh
```
