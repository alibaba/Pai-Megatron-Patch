#!/bin/bash
export NCCL_DEBUG=WARN
export LC_ALL=C.UTF-8
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=/workspace/RapidformerPro/:/workspace/RapidformerPro/Megatron-LM-main/:$PYTHONPATH

DS_CKPT_PATH=/mnt/bloom-ckpts/bloomz-1b7-optimizer-states
HF_CKPT_PATH=/mnt/bloom-ckpts/bloomz-1b7-optimizer-states-to-transformers

python convert_bloom_original_checkpoint_to_pytorch.py \
       --bloom_checkpoint_path ${DS_CKPT_PATH} \
       --pytorch_dump_folder_path ${HF_CKPT_PATH} \
       --pretraining_tp 1 \
       --bloom_config_file /mnt/bloom-ckpts/bloomz-1b7/config.json
