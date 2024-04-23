Here is the translated README, keeping everything else unchanged:

# Table of Contents
   * [Installation](#installation)
   * [Dataset & Model Download](#dataset-and-model-download)
   * [Megatron-LM-Dense Model Training Process](#Megatron-LM-Dense-Model-Training-Process)
      * [Model Format Conversion](#Megatron-LM-Dense-Model-Format-Conversion)
      * [Continue Pre-training](#Megatron-LM-Dense-Continue-Pre-training)
      * [Command Fine-tuning](#Megatron-LM-Dense-Command-Fine-tuning)
   * [Megatron-Core-Dense Model Training Process](#Megatron-Core-Dense-Model-Training-Process)
      * [Model Format Conversion](#Megatron-Core-Dense-Model-Format-Conversion)
      * [Continue Pre-training](#Megatron-Core-Dense-Continue-Pre-training)
      * [Command Fine-tuning](#Megatron-Core-Dense-Command-Fine-tuning)
   * [Downstream Task Evaluation](#Downstream-Task-Evaluation)
      * [Megatron-LM Model Format Conversion](#Megatron-LM-Dense-Model-to-Huggingface-Format)
      * [Run Evaluation Tool](#Run-Evaluation-Tool)

# Installation
It is recommended to create a container using NVIDIA's official image nvcr.io/nvidia/pytorch:23.12-py3

```bash
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
cd Pai-Megatron-Patch
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

# Dataset and Model Download
```bash
cd /mnt
mkdir llama3-ckpts
cd llama3-ckpts
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-ckpts/Meta-Llama-3-8B.tgz
tar -zxf Meta-Llama-3-8B.tgz

mkdir llama3-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/wudao_llama3bpe_content_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/wudao_llama3bpe_content_document.idx

wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/alpaca_zh-llama3-train.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/alpaca_zh-llama3-valid.json
```

# Megatron-LM-Dense Model Training Process

Run the hf2megatron_convertor.sh script, the list of parameters to be passed in is as follows
```
MEGATRON_PATH=$1                   # Path of Megatron-LM
SOURCE_CKPT_PATH=$2                # Path of original CKPT
TARGET_CKPT_PATH=$3                # Path of target CKPT
TP=$4                              # Model parallelism
PP=$5                              # Pipeline parallelism
MN=$6                              # llama3-8b 
EXTRA_VOCAB_SIZE=$7                # Vocabulary expansion size
mg2hf=$8                           # Whether to perform mg2hf conversion
```

Run the run_pretrain_megatron_llama.sh script, the list of parameters to be passed in is as follows
```
ENV=$1                          # Running environment: dlc, dsw
MEGATRON_PATCH_PATH=$2          # Set the code path of Megatron Patch
MODEL_SIZE=$3                   # Model structure parameter level: 7B, 13B
BATCH_SIZE=$4                   # Samples per card per iteration: 4, 8
GLOBAL_BATCH_SIZE=$5            # Global batch size
LR=$6                           # Learning rate: 1e-5, 5e-5
MIN_LR=$7                       # Minimum learning rate: 1e-6, 5e-6
SEQ_LEN=$8                      # Sequence length
PAD_LEN=$9                      # Padding length: 100
EXTRA_VOCAB_SIZE=${10}          # Vocabulary expansion size
PR=${11}                        # Training precision: fp16, bf16
TP=${12}                        # Model parallel

ism
PP=${13}                        # Pipeline parallelism
AC=${14}                        # Activation checkpoint mode: sel, full
DO=${15}                        # Whether to use Megatron version of Zero-1 memory optimizer: true, false
FL=${16}                        # Whether to use Flash Attention: true, false
SP=${17}                        # Whether to use sequence parallelism: true, false
TE=${18}                        # Whether to use Transformer Engine: true, false
SAVE_INTERVAL=${19}             # Interval for saving ckpt
DATASET_PATH=${20}              # Training dataset path
PRETRAIN_CHECKPOINT_PATH=${21}  # Pre-training model path
TRAIN_TOKENS=${22}              # Training token count
WARMUP_TOKENS=${23}             # Warmup token count
OUTPUT_BASEPATH=${24}           # Training output file path
```

Run the run_finetune_megatron_llama_withGA.sh script, the list of parameters to be passed in is as follows
```
ENV=$1                          # Running environment: dlc, dsw
MEGATRON_PATCH_PATH=$2          # Set the code path of Megatron Patch
MODEL_SIZE=$3                   # Model structure parameter level: 7B, 13B
BATCH_SIZE=$4                   # Samples per card per iteration: 4, 8
GLOBAL_BATCH_SIZE=$5            # Global batch size
LR=$6                           # Learning rate: 1e-5, 5e-5
MIN_LR=$7                       # Minimum learning rate: 1e-6, 5e-6
SEQ_LEN=$8                      # Sequence length
PAD_LEN=$9                      # Padding length: 100
EXTRA_VOCAB_SIZE=${10}          # Vocabulary expansion size
PR=${11}                        # Training precision: fp16, bf16
TP=${12}                        # Model parallelism
PP=${13}                        # Pipeline parallelism
AC=${14}                        # Activation checkpoint mode: sel, full
DO=${15}                        # Whether to use Megatron version of Zero-1 memory optimizer: true, false
FL=${16}                        # Whether to use Flash Attention: true, false
SP=${17}                        # Whether to use sequence parallelism: true, false
TE=${18}                        # Whether to use Transformer Engine: true, false
SAVE_INTERVAL=${19}             # Interval for saving ckpt
DATASET_PATH=${20}              # Training dataset path
VALID_DATASET_PATH=${21}        # Validation dataset path
PRETRAIN_CHECKPOINT_PATH=${22}  # Pre-training model path
TRAIN_ITERS=${23}               # Training step count
WARMUP_ITERS=${24}              # Warmup step count
OUTPUT_BASEPATH=${25}           # Training output file path
```

## Megatron-LM-Dense Model Format Conversion
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/llama
sh hf2megatron_convertor.sh \
../../../     \
/mnt/llama3-ckpts/Meta-Llama-3-8B    \
/mnt/llama3-ckpts/Meta-Llama-3-8B-to-megatron-tp4-pp1  \
4  \
1  \
llama3-8b \
0 \
false
```

## Megatron-LM-Dense Continue Pre-training
```bash
cd /workspace/Pai-Megatron-Patch/examples/llama3
sh run_pretrain_megatron_llama.sh  \
dsw  \
../../ \
8B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
256   \
bf16  \
4   \
1  \
sel  \
true   \
false  \
false   \
false   \
100000  \
/mnt/llama3-datasets/wudao_llama3bpe_content_document  \
/mnt/llama3-ckpts/Meta-Llama-3-8B-to-megatron-tp4-pp1  \
100000000   \
10000   \
/mnt/output_megatron_llama3
```

## Megatron-LM-Dense Command Fine-tuning
```bash
cd /workspace/Pai-Megatron-Patch/examples/llama3
sh run_finetune_megatron_llama_withGA.sh  \
dsw  \
../../ \
8B     \
1      \
32     \
1e-5   \
1e-6   \
128   \
128     \
256      \
bf16   \
4      \
1      \
sel    \
true   \
false  \
false  \
false \
100 \
/mnt/llama3-datasets/alpaca_zh-llama3-train.json   \
/mnt/llama3-datasets/alpaca_zh

-llama3-valid.json   \
/mnt/llama3-ckpts/Meta-Llama-3-8B-to-megatron-tp4-pp1  \
1000 \
10 \
/mnt/output_megatron_llama3/
```

# Megatron-Core-Dense Model Training Process

Run the hf2mcore_convertor.sh script, the list of parameters to be passed in is as follows
```
MODEL_SIZE=$1                  # Model parameters: 7B/13B/70B
HG_CKPT_PATH=$2                # HF's CKPT path
MEGATRON_PATH=$3               # Root directory of Megatron-LM
SOURCE_CKPT_PATH=$4            # Source path
TARGET_CKPT_PATH=$5            # Target path
TP=$6                          # Model parallelism
PP=$7                          # Pipeline parallelism
EXTRA_VOCAB_SIZE=$8            # Additional vocabulary expansion size
NUM_EXPERTS=$9                 # Number of experts
EXPERTS_TOPK=${10}             # Expert routing Topk
EP=${11}                       # Expert parallelism
mg2hf=${12}                    # Whether to perform mcore2hf conversion
```

Run the run_pretrain_mcore_llama.sh script, the list of parameters to be passed in is as follows
```
ENV=$1                          # Running environment: dlc, dsw
MEGATRON_PATCH_PATH=$2          # Set the code path of Megatron Patch
MODEL_SIZE=$3                   # Model structure parameter level: 7B, 13B
BATCH_SIZE=$4                   # Samples per card per iteration: 4, 8
GLOBAL_BATCH_SIZE=$5            # Global batch size
LR=$6                           # Learning rate: 1e-5, 5e-5
MIN_LR=$7                       # Minimum learning rate: 1e-6, 5e-6
SEQ_LEN=$8                      # Sequence length
PAD_LEN=$9                      # Padding length: 100
EXTRA_VOCAB_SIZE=${10}          # Vocabulary expansion size
PR=${11}                        # Training precision: fp16, bf16
TP=${12}                        # Model parallelism
PP=${13}                        # Pipeline parallelism
AC=${14}                        # Activation checkpoint mode: sel, full
DO=${15}                        # Whether to use Megatron version of Zero-1 memory optimizer: true, false
FL=${16}                        # Whether to use Flash Attention: true, false
SP=${17}                        # Whether to use sequence parallelism: true, false
TE=${18}                        # Whether to use Transformer Engine: true, false
MOE=${19}                       # Whether to activate MOE: true, false
SAVE_INTERVAL=${20}             # Interval for saving ckpt
DATASET_PATH=${21}              # Training dataset path
PRETRAIN_CHECKPOINT_PATH=${22}  # Pre-training model path
TRAIN_TOKENS=${23}              # Training token count
WARMUP_TOKENS=${24}             # Warmup token count
OUTPUT_BASEPATH=${25}           # Training output file path
```

Run the run_finetune_mcore_llama_withGA.sh script, the list of parameters to be passed in is as follows
```
ENV=$1                          # Running environment: dlc, dsw
MEGATRON_PATCH_PATH=$2          # Set the code path of Megatron Patch
MODEL_SIZE=$3                   # Model structure parameter level: 7B, 13B
BATCH_SIZE=$4                   # Samples per card per iteration: 4, 8
GLOBAL_BATCH_SIZE=$5            # Global batch size
LR=$6                           # Learning rate: 1e-5, 5e-5
MIN_LR=$7                       # Minimum learning rate: 1e-6, 5e-6
SEQ_LEN=$8                      # Sequence length
PAD_LEN=$9                      # Padding length: 100
EXTRA_VOCAB_SIZE=${10}          # Vocabulary expansion size
PR=${11}                        # Training precision: fp16, bf16
TP=${12}                        # Model parallelism
PP=${13}                        # Pipeline parallelism
AC=${14}                        # Activation checkpoint mode: sel, full
DO=${15}                        # Whether to use Megatron version of Zero-1 memory optimizer: true, false
FL=${16}                        # Whether to use Flash Attention: true, false
SP=${17}                        # Whether to use sequence parallelism: true, false
TE=${18}                        # Whether to use Transformer Engine: true, false
MOE=${19}                       # Whether to activate MOE: true, false
SAVE_INTERVAL=${20}             # Interval for saving ckpt
DATASET_PATH=${21}              # Training dataset path
VALID_DATASET_PATH=${22}        # Validation dataset path


PRETRAIN_CHECKPOINT_PATH=${23}  # Pre-training model path
TRAIN_ITERS=${24}               # Training step count
WARMUP_ITERS=${25}              # Warmup step count
OUTPUT_BASEPATH=${26}           # Training output file path
```

## Megatron-Core-Dense Model Format Conversion
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/llama \
sh hf2mcore_convertor.sh \
8B \
/mnt/llama3-ckpts/Meta-Llama-3-8B    \
../../../     \
/mnt/llama3-ckpts/Meta-Llama-3-8B    \
/mnt/llama3-ckpts/Meta-Llama-3-8B-to-mcore-tp4-pp1  \
4  \
1  \
256  \
0  \
0  \
0 \
false
```

## Megatron-Core-Dense Continue Pre-training
```bash
cd /workspace/Pai-Megatron-Patch/examples/llama3
sh run_pretrain_mcore_llama.sh  \
dsw  \
../../ \
8B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
256   \
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
/mnt/llama3-datasets/wudao_llama3bpe_content_document   \
/mnt/llama3-ckpts/Meta-Llama-3-8B-to-mcore-tp4-pp1  \
100000000   \
10000   \
/mnt/output_mcore_llama3
```

## Megatron-Core-Dense Command Fine-tuning
```bash
cd /workspace/Pai-Megatron-Patch/examples/llama3
sh run_finetune_mcore_llama_withGA.sh  \
dsw  \
../../ \
8B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
256   \
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
/mnt/llama3-datasets/alpaca_zh-llama3-train.json   \
/mnt/llama3-datasets/alpaca_zh-llama3-valid.json   \
/mnt/llama3-ckpts/Meta-Llama-3-8B-to-mcore-tp4-pp1  \
100000000   \
10000   \
/mnt/output_mcore_llama3
```

# Downstream Task Evaluation

## Megatron-LM-Dense Model Conversion to Huggingface Format
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/llama
sh hf2megatron_convertor.sh \
../../../     \
/mnt/llama3-ckpts/Meta-Llama-3-8B-to-mcore-tp4-pp1/release  \
/mnt/llama3-ckpts/Meta-Llama-3-8B-hf-megatron-to-hf    \
4  \
1  \
llama3-8b \
0 \
true
```

Copy the .json files (except pytorch_model.bin.index.json) from the open-source Huggingface model folder path to the /mnt/llama3-ckpts/Meta-Llama-3-8B-hf-megatron-to-hf directory to ensure the model can be used normally.

## Run Evaluation Tool
```bash
cd /workspace/Pai-Megatron-Patch/LM-Evaluation-Harness-240310
accelerate launch --main_process_port 29051 -m lm_eval \
--model hf \
--model_args pretrained=/mnt/llama3-ckpts/Meta-Llama-3-8B-hf-megatron-to-hf,trust_remote_code=True \
--tasks mmlu,ceval-valid  \
--batch_size 16
```