#!/bin/bash
set -x
ray stop
rm -rf /tmp/ray/*

# enveriment
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_num_server_call_thread=1
export VLLM_USE_RAY_SPMD_WORKER=1
export VLLM_USE_RAY_COMPILED_DAG=1
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATCH_PATH=$( dirname $( dirname ${CURRENT_DIR}))
CHATLEARN_PATH=${MEGATRON_PATCH_PATH}/backends/rl/ChatLearn
export PYTHONPATH=${MEGATRON_PATCH_PATH}:${MEGATRON_PATCH_PATH}/rl_patch/chatlearn:${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-250328:${CHATLEARN_PATH}:$PYTHONPATH
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export LOCAL_MASTER_ADDR=${MASTER_ADDR:-localhost}
ports="30000"
for i in $(seq 30001 30050); do
      ports="${ports};${i}"
done
export CUSTOM_PORTS=$ports
export num_device=$(($WORLD_SIZE * 8))

# data
export train_data_path="/mnt/data/datasets/MATH-lighteval/train.json"
export eval_data_path="/mnt/data/datasets/MATH-lighteval/test.json"
export patch_tokenizer_type=Qwen2Tokenizer
export extra_vocab_size=293
export tokenizer_path="/mnt/data/qwen-ckpts/Qwen3-8B"
export load="/mnt/data/qwen-ckpts/Qwen3-8B-to-mcore/"

# model
export max_position_embedding=40960
export policy_num_layers=36
export policy_hidden_size=4096
export policy_num_attention_heads=32
export policy_num_query_groups=8
export policy_ffn_hidden_size=12288
export tensor_model_parallel_size=4 
export pipeline_model_parallel_size=2 

# training
export final_clip_ratio=3
export clip_grad=1.0
export seed=3407
export policy_lr=2e-6
export policy_min_lr=2e-6
export eval_episode_interval=1
export save_interval=100000
export save_episode_interval=10000
export num_episode=200
export sample_per_episode=2048
export save_episode_interval=10000
export train_micro_batch_size=8
export train_global_batch_size=2048
export vllm_generation_batch_size=128
export trainer_generation_batch_size=8
export train_iters=$(( ${num_episode} * ${sample_per_episode} / ${train_global_batch_size} ))
export policy_lr_warmup_iters=0
export lr_decay_iters=160000
export max_num_batched_tokens=65536
export gpu_memory_utilization=0.85
# for qwen3 where enable_thinking
export enable_thinking=False

# vllm
export seq_length=2048
export max_new_tokens=2048
export max_seq_len_to_capture=2348
export num_inference_per_prompt=32
export policy_temperature=1.0
export policy_top_p=1.0
export policy_top_k=-1
export policy_eval_temperature=0.6
export policy_eval_top_p=0.95
export policy_eval_top_k=20

# logging and saving
export enable_tensorboard=True
export enable_wandb=False
export WANDB_API_KEY="wandb-api-key"
export exp_name=qwen3_8B_lr${policy_lr}_mbs${train_micro_batch_size}_gbs${train_global_batch_size}_tp${tensor_model_parallel_size}_pp${pipeline_model_parallel_size}_${WORLD_SIZE}nodes
export output_dir=${MEGATRON_PATCH_PATH}/output/${exp_name}
mkdir -p $output_dir/
export log_dir=${output_dir}/logs
mkdir -p $log_dir
log_file=$log_dir/${exp_name}_rank${RANK}.log
export tensorboard_dir=${output_dir}/tensorboard
export wandb_dir=${output_dir}
export save_dir=${output_dir}

python ../qwen2_5/train_grpo_qwen.py -c ${CURRENT_DIR}/configs/grpo/grpo.yaml 2>&1 | tee ${log_file} ; exit ${PIPESTATUS[0]}