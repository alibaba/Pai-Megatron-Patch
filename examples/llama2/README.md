## Megatron训练流程
### 模型格式转换
使用我们提供的模型转换脚本，将huggingface格式的模型文件转换为megatron格式：
```bash
cd /mnt
mkdir llama2-ckpts
cd llama2-ckpts
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama2-ckpts/Llama-2-13b-hf.tgz
tar -zxf Llama-2-13b-hf.tgz

cd /workspace/PAI-Megatron-Patch/toolkits/model_checkpoints_convertor/llama
sh model_convertor.sh \
../../../     \
/mnt/llama2-ckpts/Llama-2-13b-hf     \
/mnt/llama2-ckpts/Llama-2-13b-hf-to-megatron-tp4-pp1  \
4  \
1  \
llama2-13b \
0 \
false
```

### 继续预训练
运行run_pretrain_megatron_llama.sh脚本，需要传入的参数列表如下
```
ENV=$1                          # 运行环境: dlc, dsw
MEGATRON_PATCH_PATH=$2          # 设置Megatron Patch的代码路径
MODEL_SIZE=$3                   # 模型结构参数量级：7B, 13B
BATCH_SIZE=$4                   # 每卡训练一次迭代样本数: 4, 8
GLOBAL_BATCH_SIZE=$5            # 全局batch size
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
TE=${18}                        # 是否使用Transformer Engine: true, false
SAVE_INTERVAL=${19}             # 保存ckpt的间隔
DATASET_PATH=${20}              # 训练数据集路径
PRETRAIN_CHECKPOINT_PATH=${21}  # 预训练模型路径
TRAIN_TOKENS=${22}              # 训练token数
WARMUP_TOKENS=${23}             # 预热token数
OUTPUT_BASEPATH=${24}           # 训练输出文件路径
```
单机运行示例如下：
```bash
cd /workspace/PAI-Megatron-Patch/examples/llama2
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama2-datasets/wudao_llamabpe_text_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama2-datasets/wudao_llamabpe_text_document.idx
sh run_pretrain_megatron_llama.sh  \
dsw  \
../../ \
13B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
0   \
bf16  \
4   \
1  \
sel  \
true   \
false  \
false   \
false   \
100000  \
wudao_llamabpe_text_document   \
/mnt/llama2-ckpts/Llama-2-13b-hf-to-megatron-tp4-pp1   \
100000000   \
10000   \
/mnt/output_megatron_llama2
```

### 有监督微调
运行run_finetune_megatron_llama.sh脚本，需要传入的参数列表如下
```
ENV=$1                          # 运行环境: dlc, dsw
MEGATRON_PATCH_PATH=$2          # 设置Megatron Patch的代码路径
MODEL_SIZE=$3                   # 模型结构参数量级: 7B, 13B
BATCH_SIZE=$4                   # 每卡训练一次迭代样本数: 4, 8
LR=$5                           # 学习率: 1e-5, 5e-5
MIN_LR=$6                       # 最小学习率: 1e-6, 5e-6
SEQ_LEN=$7                      # 序列长度
PAD_LEN=$8                      # Padding长度：100
EXTRA_VOCAB_SIZE=$9          # 词表扩充大小
PR=${10}                        # 训练精度: fp16, bf16
TP=${11}                        # 模型并行度
PP=${12}                        # 流水并行度
AC=${13}                        # 激活检查点模式: sel, full
DO=${14}                        # 是否使用Megatron版Zero-1降显存优化器: true, false
FL=${15}                        # 是否使用Flash Attention: true, false
SP=${16}                        # 是否使用序列并行: true, false
TE=${17}                        # 是否使用Transformer engine: true, false
TRAIN_DATASET_PATH=${18}        # 训练数据集路径
VALID_DATASET_PATH=${19}        # 验证数据集路径
PRETRAIN_CHECKPOINT_PATH=${20}  # 预训练模型路径
EPOCH=${21}                     # 训练迭代轮次
OUTPUT_BASEPATH=${22}           # 训练输出文件路径
```
DSW单机运行示例如下：
```bash
cd /workspace/PAI-Megatron-Patch/examples/llama2
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama2-datasets/wudao_train.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama2-datasets/wudao_valid.json
sh run_finetune_megatron_llama.sh  \
dsw  \
../../ \
13B     \
1      \
1e-5   \
1e-6   \
128   \
128     \
0      \
bf16   \
4      \
1      \
sel    \
true   \
false  \
false  \
false \
wudao_train.json   \
wudao_valid.json   \
/mnt/llama2-ckpts/Llama-2-13b-hf-to-megatron-tp4-pp1   \
2   \
/mnt/output_megatron_llama2/
```

### 带Global BS和Gradient Accumulation的有监督微调

```bash
cd /workspace/PAI-Megatron-Patch/examples/llama2
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama2-datasets/alpaca_zh-llama2-train.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama2-datasets/alpaca_zh-llama2-valid.json
sh run_finetune_megatron_llama_withGA.sh  \
dsw  \
../../ \
13B     \
1      \
32     \
1e-5   \
1e-6   \
128   \
128     \
0      \
bf16   \
4      \
1      \
sel    \
true   \
false  \
false  \
false \
100 \
alpaca_zh-llama2-train.json   \
alpaca_zh-llama2-valid.json   \
/mnt/llama2-ckpts/Llama-2-13b-hf-to-megatron-tp4-pp1   \
1000 \
10 \
/mnt/output_megatron_llama2/
```