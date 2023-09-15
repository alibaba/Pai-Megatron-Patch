## Megatron推理
此处复用了Megatron-LM中的推理框架。  
改动：  
- 修改tokenizer处理数据的部分，适配huggingface的tokenizer  
- 增加推理过程中对重复生成的处理，支持repetition_penalty.  


## 模型推理示例
对于Megatron-LM训练的模型，可以直接用Megatron-LM框架进行推理。
参数如下
```bash
ENV=$1                          # 运行环境: dlc, dsw
MEGATRON_PATH=$2                # 设置开源Megatron的代码路径
MEGATRON_PATCH_PATH=$3          # 设置Megatron Patch的代码路径
CHECKPOINT_PATH=$4              # 模型微调阶段的模型保存路径
MODEL_SIZE=$5                   # 模型结构参数量级: 1.1B, 1.7B, 7.1B
TP=$6                           # 模型并行度
BS=$7                           # 每卡推理一次迭代样本数: 1, 4, 8
SEQ_LEN=$8						# 序列长度: 256, 512, 1024
PAD_LEN=$9                      # PAD长度：需要将文本拼接到的长度
EXTRA_VOCAB_SIZE=${10}          # 模型转换时增加的token数量
PR=${11}                        # 推理采用的精度: fp16, bf16
TOP_K=${12}                     # 采样策略中选择排在前面的候选词数量(0-n): 0, 5, 10, 20
INPUT_SEQ_LEN=${13}             # 输入序列长度: 512
OUTPUT_SEQ_LEN=${14}            # 输出序列长度: 256
INPUT_FILE=${15}                # 需要推理的文本文件: input.txt, 每行为一个样本
OUTPUT_FILE=${16}               # 推理输出的文件: output.txt
# TOP_K和TOP_P必须有一个为0
TOP_P=${17}                     # 采样策略中选择排在前面的候选词百分比(0-1): 0, 0.85, 0.95
TEMPERATURE=${18}               # 采样策略中温度惩罚: 1-n
REPETITION_PENALTY=${19}        # 避免生成是产生大量重复，可以设置为(1-2)默认为1.2
```
运行以下命令进行模型推理。  

以下有监督微调过程保存模型的推理代码，需要将run_text_generation_megatron_llama.sh脚本中CUDA_VISIBLE_DEVICES参数设置为0；GPUS_PER_NODE参数设置为1；同时使用下列代码进行推理。此时使用单卡进行推理。注意：此处模型tp为1，可使用单卡推理；如果tp>1，则需使用相应卡数进行推理。
```bash
export WORK_DIR=/mnt/workspace
cd ${WORK_DIR}/PAI-Megatron-Patch/examples/llama2
bash run_text_generation_megatron_llama.sh \
dsw \
/root/Megatron-LM-23.04 \
${WORK_DIR}/PAI-Megatron-Patch \
../../../llama2-train \
7B \
1 \
1 \
1024 \
1024 \
0 \
fp16 \
10 \
512 \
512 \
${WORK_DIR}/pred_input.jsonl \
${WORK_DIR}/llama2_pred.txt \
0 \
1.0 \
1.2
```

