#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

options=" \
        --model hf \
        --model_args pretrained=/mnt/qwen-ckpts/Qwen-1_8B,trust_remote_code=true \
        --tasks cmmlu,ceval-valid \
        --batch_size 16
        "

run_cmd="accelerate launch -m lm_eval ${options}"

echo ${run_cmd}
eval ${run_cmd}