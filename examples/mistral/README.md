# Table of Contents

   * [Installation](#installation)
   * [Dataset and Model Download](#dataset-and-model-download)
   * [Megatron-LM-Dense Model Training Process](#megatron-lm-dense-model-training-process)
      * [Megatron-LM-Dense Model Format Conversion](#megatron-lm-dense-model-format-conversion)
      * [Continue Pretraining Megatron-LM-Dense](#continue-pretraining-megatron-lm-dense)
      * [Instruction Fine-tuning Megatron-LM-Dense](#instruction-fine-tuning-megatron-lm-dense)
   * [Megatron-Core-Dense Model Training Process](#megatron-core-dense-model-training-process)
      * [Megatron-Core-Dense Model Format Conversion](#megatron-core-dense-model-format-conversion)
      * [Continue Pretraining Megatron-Core-Dense](#continue-pretraining-megatron-core-dense)
      * [Instruction Fine-tuning Megatron-Core-Dense](#instruction-fine-tuning-megatron-core-dense)
   * [Megatron-Core-MoE Model Training Process](#megatron-core-moe-model-training-process)
      * [Megatron-Core-MoE Model Format Conversion](#megatron-core-moe-model-format-conversion)
      * [Continue Pretraining Megatron-Core-MoE](#continue-pretraining-megatron-core-moe)
      * [Instruction Fine-tuning Megatron-Core-MoE](#instruction-fine-tuning-megatron-core-moe)
   * [Downstream Task Evaluation](#downstream-task-evaluation)
      * [Megatron-LM-Dense Model Conversion to Huggingface Format](#megatron-lm-dense-model-conversion-to-huggingface-format)
      * [Run Evaluation Tools](#run-evaluation-tools)


# Installation
It is recommended to create a container using the official NVIDIA image nvcr.io/nvidia/pytorch:23.12-py3

```bash
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
cd Pai-Megatron-Patch
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

# Dataset and Model Download
```bash
cd /mnt
mkdir mistral-ckpts
cd mistral-ckpts
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-ckpts/Mistral-7B-v0.1.tgz
tar -zxf Mistral-7B-v0.1.tgz

mkdir mistral-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/wudao_mistralbpe_content_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/wudao_mistralbpe_content_document.idx

wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/alpaca_zh-mistral-train.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/mistral-datasets/alpaca_zh-mistral-valid.json

```

# Megatron-LM-Dense Model Training Process

Run the hf2megatron_convertor.sh script, the list of required parameters is as follows
```
MEGATRON_PATH=$1                   # Path to Megatron-LM
SOURCE_CKPT_PATH=$2                # Original CKPT path
TARGET_CKPT_PATH=$3                # Target CKPT path
TP=$4                              # Model parallelism
PP=$5                              # Pipeline parallelism
MN=$6                              # mistral-7b
EXTRA_VOCAB_SIZE=$7                # Extra vocabulary size
mg2hf=$8                           # Whether to perform mg2hf conversion
```

Run the run_pretrain_megatron_mistral.sh script, the list of required parameters is as follows

```
ENV=$1                          # Running environment: dlc, dsw
MEGATRON_PATCH_PATH=$2          # Set the path to Megatron Patch code
MODEL_SIZE=$3                   # Model structure parameter level: 7B, 13B
BATCH_SIZE=$4                   # Number of samples per card per iteration: 4, 8
GLOBAL_BATCH_SIZE=$5            # Global batch size
LR=$6                           # Learning rate: 1

e-5, 5e-5
MIN_LR=$7                       # Minimum learning rate: 1e-6, 5e-6
SEQ_LEN=$8                      # Sequence length
PAD_LEN=$9                      # Padding length: 100
EXTRA_VOCAB_SIZE=${10}          # Extra vocabulary size
PR=${11}                        # Training precision: fp16, bf16
TP=${12}                        # Model parallelism
PP=${13}                        # Pipeline parallelism
AC=${14}                        # Activation checkpoint mode: sel, full
DO=${15}                        # Whether to use Megatron version Zero-1 memory reduction optimizer: true, false
FL=${16}                        # Whether to use Flash Attention: true, false
SP=${17}                        # Whether to use sequence parallelism: true, false
TE=${18}                        # Whether to use Transformer Engine: true, false
SAVE_INTERVAL=${19}             # Save checkpoint interval
DATASET_PATH=${20}              # Training dataset path
PRETRAIN_CHECKPOINT_PATH=${21}  # Pre-training model path
TRAIN_TOKENS=${22}              # Training token number
WARMUP_TOKENS=${23}             # Warmup token number
OUTPUT_BASEPATH=${24}           # Training output file path
```

Run the run_finetune_megatron_mistral_withGA.sh script, the list of required parameters is as follows

```
ENV=$1                          # Running environment: dlc, dsw
MEGATRON_PATCH_PATH=$2          # Set the path to Megatron Patch code
MODEL_SIZE=$3                   # Model structure parameter level: 7B, 13B
BATCH_SIZE=$4                   # Number of samples per card per iteration: 4, 8
GLOBAL_BATCH_SIZE=$5            # Global batch size
LR=$6                           # Learning rate: 1e-5, 5e-5
MIN_LR=$7                       # Minimum learning rate: 1e-6, 5e-6
SEQ_LEN=$8                      # Sequence length
PAD_LEN=$9                      # Padding length: 100
EXTRA_VOCAB_SIZE=${10}          # Extra vocabulary size
PR=${11}                        # Training precision: fp16, bf16
TP=${12}                        # Model parallelism
PP=${13}                        # Pipeline parallelism
AC=${14}                        # Activation checkpoint mode: sel, full
DO=${15}                        # Whether to use Megatron version Zero-1 memory reduction optimizer: true, false
FL=${16}                        # Whether to use Flash Attention: true, false
SP=${17}                        # Whether to use sequence parallelism: true, false
TE=${18}                        # Whether to use Transformer Engine: true, false
SAVE_INTERVAL=${19}             # Save checkpoint interval
DATASET_PATH=${20}              # Training dataset path
VALID_DATASET_PATH=${21}        # Validation dataset path
PRETRAIN_CHECKPOINT_PATH=${22}  # Pre-training model path
TRAIN_ITERS=${23}               # Training step number
WARMUP_ITERS=${24}              # Warmup step number
OUTPUT_BASEPATH=${25}           # Training output file path
```

## Megatron-LM-Dense Model Format Conversion
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/mistral
sh hf2megatron_convertor.sh \
../../../     \
/mnt/mistral-ckpts/Mistral-7B-v0.1    \
/mnt/mistral-ckpts/Mistral-7B-v0.1-hf-to-megatron-tp4-pp1  \
4  \
1  \
mistral-7b \
0 \
false
```

## Continue Pretraining Megatron-LM-Dense
```bash
cd /workspace/Pai-Megatron-Patch/examples/mistral
sh run_pretrain_megatron_mistral.sh  \
dsw  \
../../ \
7B   \
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
/mnt/mistral-datasets/wudao_mistralbpe_content_document   \
/mnt/mistral-ckpts/Mistral-7B-v0.1-hf-to-megatron-tp4-pp1  \
100000000   \
10000   \
/mnt/output_megatron_mistral
```

## Instruction Fine-tuning Megatron-LM-Dense
```bash
cd /workspace/Pai-Megatron-Patch/examples/mistral
sh run_finetune_megatron_mistral_withGA.sh  \
dsw  \
../../ \
7B    

 \
1      \
8      \
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
100000  \
/mnt/mistral-datasets/alpaca_zh-mistral-train.json   \
/mnt/mistral-datasets/alpaca_zh-mistral-valid.json   \
/mnt/mistral-ckpts/Mistral-7B-v0.1-hf-to-megatron-tp4-pp1   \
100000000   \
10000   \
/mnt/output_megatron_mistral
```

# Megatron-Core-Dense Model Training Process
Run the hf2mcore_convertor.sh script, the list of required parameters is as follows
```
MODEL_SIZE=$1                  # Model parameters: 7B/8x7B
HG_CKPT_PATH=$2                # HF CKPT path
MEGATRON_PATH=$3               # Megatron-LM root directory
SOURCE_CKPT_PATH=$4            # Source path
TARGET_CKPT_PATH=$5            # Target path
TP=$6                          # Model parallelism
PP=$7                          # Pipeline parallelism
EXTRA_VOCAB_SIZE=$8            # Extra vocabulary size
NUM_EXPERTS=$9                 # Number of experts
EXPERTS_TOPK=${10}             # Expert routing Topk
EP=${11}                       # Expert parallelism
mg2hf=${12}                    # Whether to perform mcore2hf conversion
WS=${13}                       # When 8x7B, specify world size
```

Run the run_pretrain_mcore_mistral.sh script, the list of required parameters is as follows
```
ENV=$1                          # Running environment: dlc, dsw
MEGATRON_PATCH_PATH=$2          # Set the path to Megatron Patch code
MODEL_SIZE=$3                   # Model structure parameter level: 7B, 13B
BATCH_SIZE=$4                   # Number of samples per card per iteration: 4, 8
GLOBAL_BATCH_SIZE=$5            # Global batch size
LR=$6                           # Learning rate: 1e-5, 5e-5
MIN_LR=$7                       # Minimum learning rate: 1e-6, 5e-6
SEQ_LEN=$8                      # Sequence length
PAD_LEN=$9                      # Padding length: 100
EXTRA_VOCAB_SIZE=${10}          # Extra vocabulary size
PR=${11}                        # Training precision: fp16, bf16
TP=${12}                        # Model parallelism
PP=${13}                        # Pipeline parallelism
AC=${14}                        # Activation checkpoint mode: sel, full
DO=${15}                        # Whether to use Megatron version Zero-1 memory reduction optimizer: true, false
FL=${16}                        # Whether to use Flash Attention: true, false
SP=${17}                        # Whether to use sequence parallelism: true, false
TE=${18}                        # Whether to use Transformer Engine: true, false
MOE=${19}                       # Whether to enable MOE: true, false
SAVE_INTERVAL=${20}             # Save checkpoint interval
DATASET_PATH=${21}              # Training dataset path
PRETRAIN_CHECKPOINT_PATH=${22}  # Pre-training model path
TRAIN_TOKENS=${23}              # Training token number
WARMUP_TOKENS=${24}             # Warmup token number
OUTPUT_BASEPATH=${25}           # Training output file path
```

Run the run_finetune_mcore_mistral_withGA.sh script, the list of required parameters is as follows
```
ENV=$1                          # Running environment: dlc, dsw
MEGATRON_PATCH_PATH=$2          # Set the path to Megatron Patch code
MODEL_SIZE=$3                   # Model structure parameter level: 7B, 13B
BATCH_SIZE=$4                   # Number of samples per card per iteration: 4, 8
GLOBAL_BATCH_SIZE=$5            # Global batch size
LR=$6                           # Learning rate: 1e-5, 5e-5
MIN_LR=$7                       # Minimum learning rate: 1e-6, 5e-6
SEQ_LEN=$8                      # Sequence length
PAD_LEN=$9                      # Padding length: 100
EXTRA_VOCAB_SIZE=${10}          # Extra vocabulary size
PR=${11}                        # Training precision: fp16, bf16
TP=${12}                        # Model parallelism
PP=${13}                        # Pipeline parallelism
AC=${14}                        # Activation checkpoint mode: sel, full
DO=${15}                        # Whether to use Megatron version Zero-1 memory reduction optimizer: true, false
FL=${

16}                        # Whether to use Flash Attention: true, false
SP=${17}                        # Whether to use sequence parallelism: true, false
TE=${18}                        # Whether to use Transformer Engine: true, false
MOE=${19}                       # Whether to enable MOE: true, false
SAVE_INTERVAL=${20}             # Save checkpoint interval
DATASET_PATH=${21}              # Training dataset path
VALID_DATASET_PATH=${22}        # Validation dataset path
PRETRAIN_CHECKPOINT_PATH=${23}  # Pre-training model path
TRAIN_ITERS=${24}               # Training step number
WARMUP_ITERS=${25}              # Warmup step number
OUTPUT_BASEPATH=${26}           # Training output file path
```

## Megatron-Core-Dense Model Format Conversion
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/mistral \
sh hf2mcore_convertor.sh \
7B \
/mnt/mistral-ckpts/Mistral-7B-v0.1 \
../../../     \
/mnt/mistral-ckpts/Mistral-7B-v0.1 \
/mnt/mistral-ckpts/Mistral-7B-v0.1-to-mcore-tp4-pp1  \
4  \
1  \
0  \
0  \
0  \
0 \
false
```

## Continue Pretraining Megatron-Core-Dense
```bash
cd /workspace/Pai-Megatron-Patch/examples/mistral
sh run_pretrain_mcore_mistral.sh  \
dsw  \
../../ \
7B   \
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
false \
100000  \
/mnt/mistral-datasets/wudao_mistralbpe_content_document \
/mnt/mistral-ckpts/Mistral-7B-v0.1-to-mcore-tp4-pp1   \
100000000   \
10000   \
/mnt/output_mcore_mistral
```

## Instruction Fine-tuning Megatron-Core-Dense
```bash
cd /workspace/Pai-Megatron-Patch/examples/mistral
sh run_finetune_mcore_mistral_withGA.sh  \
dsw  \
../../ \
7B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
0   \
bf16  \
4  \
1  \
sel  \
true   \
false  \
false   \
false   \
false \
100000  \
/mnt/mistral-datasets/alpaca_zh-mistral-train.json   \
/mnt/mistral-datasets/alpaca_zh-mistral-valid.json   \
/mnt/mistral-ckpts/Mistral-7B-v0.1-to-mcore-tp4-pp1   \
100000000   \
10000   \
/mnt/output_mcore_mistral
```

# Megatron-Core-MoE Model Training Process

## Megatron-Core-MoE Model Format Conversion
Based on Sparse-Upcycled Dense to MoE model format conversion
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/mistral \
sh hf2mcore_convertor.sh \
7B \
/mnt/mistral-ckpts/Mistral-7B-v0.1 \
../../../     \
/mnt/mistral-ckpts/Mistral-7B-v0.1 \
/mnt/mistral-ckpts/Mistral-7B-v0.1-to-mcore-tp4-pp1-ep1-exp2 \
4  \
1  \
0  \
2  \
2  \
1 \
false
```

Direct conversion of Mixtral-8x7B model to Mcore format
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/mistral \
sh hf2mcore_convertor.sh \
8x7B \
/mnt/mistral-ckpts/Mixtral-8x7B-v0.1 \
../../../     \
/mnt/mistral-ckpts/Mixtral-8x7B-v0.1 \
/mnt/mistral-ckpts/Mixtral-8x7B-v0.1-to-mcore-tp4-pp1-ep4-exp8-ws16 \
4  \
1  \
0  \
8  \
2  \
4 \
false \
16
```

## Continue Pretraining Megatron-Core-MoE
```bash
cd /workspace/Pai-Megatron-Patch/examples/mistral
sh run_pretrain_mcore_mistral.sh  \
dsw  \
../../ \
7B   \
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
true \
100000  \
/mnt/mistral-datasets/wudao_mistralbpe_content_document \
/mnt/mistral-ckpts/Mistral-7B-v0.1-to-mcore-tp4-pp1-ep1-exp2 \
100000000   \
10000   \
/mnt/output_mcore_mistral
```

## Instruction Fine-tuning Megatron-Core-MoE
```bash
cd /workspace/Pai-Megatron-Patch/examples/mistral
sh run_finetune_mcore_mistral_withGA.sh  \
dsw  \
../../ \
7B   \
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
true \
100000  \
/mnt/mistral-datasets/alpaca_zh-mistral-train.json   \
/mnt/mistral-datasets/alpaca_zh-mistral-valid.json   \
/mnt/mistral-ckpts/Mistral-7B-v0.1-to-mcore-tp4-pp1-ep1-exp2 \
100000000   \
10000   \
/mnt/output_mcore_mistral
```

# Downstream Task Evaluation

## Megatron-LM-Dense Model Conversion to Huggingface Format
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/mistral
sh hf2megatron_convertor.sh \
../../../     \
/mnt/mistral-ckpts/Mistral-7B-v0.1-hf-to-megatron-tp4-pp1/release  \
/mnt/mistral-ckpts/Mistral-7B-v0.1-megatron-to-hf    \
4  \
1  \
mistral-7b \
0 \
true
```

## Run Evaluation Tools
```bash
cd /workspace/Pai-Megatron-Patch/LM-Evaluation-Harness-240310
accelerate launch --main_process_port 29051 -m lm_eval \
--model hf \
--model_args pretrained=/mnt/mistral-ckpts/Mistral-7B-v0.1-megatron-to-hf,trust_remote_code=True \
--tasks mmlu,ceval-valid  \
--batch_size 16
```

