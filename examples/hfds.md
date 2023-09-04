## Huggingface&DeepSpeed训练流程
### 有监督微调/继续预训练

运行run_ds_train_huggingface_llama.sh脚本，需要传入的参数列表如下：
```bash
ENV=$1                             # 运行环境配置：dsw,dlc
MODEL_SIZE=$2                      # 模型结构参数量级: 7B,13B
BATCH_SIZE=$3                      # 每卡训练一次迭代样本数: 4, 8
GA_STEPS=$4                        # 梯度累积step数
LR=$5                              # 学习率: 1e-5, 5e-5
SEQ_LEN=$6                         # 序列长度: 2048
PR=$7                              # 训练精度: fp16, bf16
ZERO=$8                            # DeepSpeed ZERO降显存: 1,2,3
GC=$9                              # 是否使用gradient-checkpointing: true, false
TRAIN_DATASET_PATH=${10}           # 训练集路径, 支持单一文件或者文件夹形式输入
VALID_DATASET_PATH=${11}           # 验证集路径, 支持单一文件或者文件夹形式输入
PRETRAIN_CHECKPOINT_PATH=${12}     # 预训练模型路径
EPOCH=${13}                        # 训练epoch数
OUTPUT_BASEPATH=${14}              # 训练输出文件路径
```
单机运行示例如下：
```bash
cd /mnt/workspace
mkdir test_llama2_hf
cd test_llama2_hf
export WORK_DIR=/mnt/workspace/
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama2-codes/ds_config_TEMPLATE.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama2-codes/ds_train_huggingface_llama.py
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama2-codes/run_ds_train_huggingface_llama.sh
bash run_ds_train_huggingface_llama.sh \
dsw \
7B \
1 \
2 \
1e-5 \
2048 \
bf16 \
2 \
true \
${WORK_DIR}/llama2-datasets/wudao_train.json \
${WORK_DIR}/llama2-datasets/wudao_valid.json \
${WORK_DIR}/llama2-ckpts/Llama-2-7b-hf \
2 \
${WORK_DIR}/output_llama2
```
