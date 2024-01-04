## Megatron训练流程
### 模型格式转换
使用我们提供的模型转换脚本，将huggingface格式的模型文件转换为megatron格式：
```bash
cd /mnt/workspace/
mkdir llama2-ckpts
cd llama2-ckpts
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama2-ckpts/Llama-2-7b-hf.tgz
tar -zxf Llama-2-7b-hf.tgz
mv Llama-2-7b-hf llama2-7b-hf

cd /mnt/workspace/PAI-Megatron-Patch/toolkits/model_checkpoints_convertor/llama
sh model_convertor.sh \
/root/Megatron-LM-230512        \
/mnt/workspace/llama2-ckpts/llama2-7b-hf         \
/mnt/workspace/llama2-ckpts/llama2-7b-hf-to-megatron-tp1-pp1  \
1  \
1  \
llama-7b \
0 \
false
```

为方便用户试用，我们也提供了转好格式的模型，可直接下载使用：
```bash
cd /mnt/workspace/
mkdir llama2-ckpts
cd llama2-ckpts
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama2-ckpts/llama-2-7b-hf-to-megatron-tp1-pp1.tgz
tar -zxf llama-2-7b-hf-to-megatron-tp1-pp1.tgz
```
### 继续预训练
运行run_pretrain_megatron_llama.sh脚本，需要传入的参数列表如下
```
ENV=$1                          # 运行环境: dlc, dsw
MEGATRON_PATH=$2                # 设置开源Megatron的代码路径
MEGATRON_PATCH_PATH=$3          # 设置Megatron Patch的代码路径
MODEL_SIZE=$4                   # 模型结构参数量级：7B, 13B
BATCH_SIZE=$5                   # 每卡训练一次迭代样本数: 4, 8
GLOBAL_BATCH_SIZE=$6            # 全局batch size
LR=$7                           # 学习率: 1e-5, 5e-5
MIN_LR=$8                       # 最小学习率: 1e-6, 5e-6
SEQ_LEN=$9                      # 序列长度
PAD_LEN=${10}                   # Padding长度：100
EXTRA_VOCAB_SIZE=${11}          # 词表扩充大小
PR=${12}                        # 训练精度: fp16, bf16
TP=${13}                        # 模型并行度
PP=${14}                        # 流水并行度
AC=${15}                        # 激活检查点模式: sel, full
DO=${16}                        # 是否使用Megatron版Zero-1降显存优化器: true, false
FL=${17}                        # 是否使用Flash Attention: true, false
SP=${18}                        # 是否使用序列并行: true, false
SAVE_INTERVAL=${19}             # 保存ckpt的间隔
DATASET_PATH=${20}              # 训练数据集路径
PRETRAIN_CHECKPOINT_PATH=${21}  # 预训练模型路径
TRAIN_TOKENS=${22}              # 训练token数
WARMUP_TOKENS=${23}             # 预热token数
OUTPUT_BASEPATH=${24}           # 训练输出文件路径
```
单机运行示例如下：
```bash
export WORK_DIR=/mnt/workspace
cd ${WORK_DIR}/PAI-Megatron-Patch/examples/llama2
sh run_pretrain_megatron_llama.sh  \
dsw  \
/root/Megatron-LM-230512   \
${WORK_DIR}/PAI-Megatron-Patch  \
7B   \
1    \
8 \
1e-5   \
1e-6   \
2048  \
80  \
0   \
fp16  \
1   \
1  \
sel  \
true   \
false  \
false   \
100000  \
${WORK_DIR}/llama2-datasets/wudao/wudao_llamabpe_text_document   \
${WORK_DIR}/llama2-ckpts/llama2-7b-hf-to-megatron-tp1-pp1   \
100000000   \
10000   \
${WORK_DIR}/output_megatron_llama2/
```

### 有监督微调
运行run_finetune_megatron_llama.sh脚本，需要传入的参数列表如下
```
ENV=$1                          # 运行环境: dlc, dsw
MEGATRON_PATH=$2                # 设置开源Megatron的代码路径
MEGATRON_PATCH_PATH=$3          # 设置Megatron Patch的代码路径
MODEL_SIZE=$4                   # 模型结构参数量级: 7B, 13B
BATCH_SIZE=$5                   # 每卡训练一次迭代样本数: 4, 8
LR=$6                           # 学习率: 1e-5, 5e-5
MIN_LR=$7                       # 最小学习率: 1e-6, 5e-6
SEQ_LEN=$8                      # 序列长度
PAD_LEN=$9                      # Padding长度：100
EXTRA_VOCAB_SIZE=${10}          # 词表扩充大小
PR=${11}                        # 训练精度: fp16, bf16
TP=${12}                        # 模型并行度
PP=${13}                        # 流水并行度
AC=${14}                        # 激活检查点模式: sel, full
DO=${15}                        # 是否使用Megatron版Zero-1降显存优化器: true, false
FL=${16}                        # 是否使用Flash Attention: true, false
SP=${17}                        # 是否使用序列并行: true, false
TRAIN_DATASET_PATH=${18}        # 训练数据集路径
VALID_DATASET_PATH=${19}        # 验证数据集路径
PRETRAIN_CHECKPOINT_PATH=${20}  # 预训练模型路径
EPOCH=${21}                     # 训练迭代轮次
OUTPUT_BASEPATH=${22}           # 训练输出文件路径
```
DSW单机运行示例如下：
```bash
export WORK_DIR=/mnt/workspace
cd ${WORK_DIR}/PAI-Megatron-Patch/examples/llama2
sh run_finetune_megatron_llama.sh  \
dsw  \
/root/Megatron-LM-230512  \
${WORK_DIR}/PAI-Megatron-Patch  \
7B     \
1      \
1e-5   \
1e-6   \
2048   \
80     \
0      \
bf16   \
1      \
1      \
sel    \
true   \
false  \
false  \
${WORK_DIR}/llama2-datasets/wudao_train.json   \
${WORK_DIR}/llama2-datasets/wudao_valid.json   \
${WORK_DIR}/llama2-ckpts/llama2-7b-hf-to-megatron-tp1-pp1   \
2   \
${WORK_DIR}/output_megatron_llama2/
```
