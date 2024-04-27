#!/bin/bash
python toolkits/model_checkpoints_convertor/mistral/hf2mcore_mixtral.py  --load_path  /workspace/checkpoints/teeny-tiny-mixtral --save_path /workspace/checkpoints/teeny-tiny-mixtral-megatrone  --target_expert_model_parallel_size 2   \
    --target_tensor_model_parallel_size 2 \
    --target_pipeline_model_parallel_size 1 \
    --target_params_dtype bf16 \
    --world_size 4