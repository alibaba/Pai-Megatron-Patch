#!/bin/bash
export NCCL_DEBUG=WARN
export LC_ALL=C.UTF-8
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=/workspace/RapidformerPro/:/workspace/RapidformerPro/Megatron-LM-main/:$PYTHONPATH

DS_CKPT_PATH=/mnt/bloom-ckpts/bloomz-1b7-optimizer-states
MG_CKPT_PATH=/mnt/bloom-ckpts/bloomz-1b7-optimizer-states-to-megatron

python deepspeed_to_megatron.py \
       --input_folder ${DS_CKPT_PATH} \
       --output_folder ${MG_CKPT_PATH} \
       --target_tp 1 \
       --target_pp 1 \
       --for_release
