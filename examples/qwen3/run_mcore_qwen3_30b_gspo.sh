#!/bin/bash
set -x

ray stop
rm -rf /tmp/ray/*

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export RAY_DEDUP_LOGS=1
export WORLD_SIZE=${WORLD_SIZE:-1}
export GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU:-$(python -c "import torch; print(torch.cuda.device_count())")}
export RANK=${RANK:-0}
export LOCAL_MASTER_ADDR=${MASTER_ADDR:-localhost}
ports="30000"
for i in $(seq 30001 30050); do
      ports="${ports};${i}"
done
export CUSTOM_PORTS=$ports
export num_device=$(($WORLD_SIZE * $GPUS_PER_NODE))

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATCH_PATH=$( dirname $( dirname ${CURRENT_DIR}))
CHATLEARN_ROOT_PATH=${MEGATRON_PATCH_PATH}/backends/rl/ChatLearn
CHATLEARN_KERNEL_PATH=${MEGATRON_PATCH_PATH}/backends/rl/ChatLearn/chatlearn
export PYTHONPATH=${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-250624:${CHATLEARN_ROOT_PATH}:${CHATLEARN_KERNEL_PATH}:$PYTHONPATH

export RAY_CGRAPH_get_timeout=200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_num_server_call_thread=1
export RAY_DEDUP_LOGS=0
export VLLM_USE_RAY_SPMD_WORKER=1
export VLLM_USE_RAY_COMPILED_DAG=1

hf_ckpt_path=/mnt/data/ckpts/huggingface/Qwen3-30B-A3B
mcore_ckpt_path=/mnt/data/ckpts/mcore/Qwen3-30B-A3B-to-mcore
exp_name="test_qwen3_30b"
export output_dir=${MEGATRON_PATCH_PATH}/output/${exp_name}
mkdir -p $output_dir/
export log_dir=${output_dir}/logs
mkdir -p $log_dir
log_file=$log_dir/${exp_name}_rank${RANK}.log

python entrypoint.py grpo --config-file configs/grpo_megatron.yaml \
        runtime_args.exp_name=${exp_name} \
        runtime_args.log_args_dict.enable_tensorboard=True \
        runtime_args.train_backend=megatron \
        runtime_args.data_path=/mnt/data/datasets/MATH-lighteval/train.json \
        runtime_args.eval_data_path=/mnt/data/datasets/MATH-lighteval/test.json \
        runtime_args.output_dir=${output_dir}\
        runtime_args.num_episode=100 \
        runtime_args.sample_per_episode=1024 \
        runtime_args.train_global_batch_size=256 \
        runtime_args.train_micro_batch_size=1 \
        runtime_args.save_episode_interval=1000000 \
        runtime_args.log_args_dict.enable_tensorboard=true \
        runtime_args.log_args_dict.tensorboard_dir=${output_dir}/tensorboard \
        runtime_args.eval_episode_interval=1 \
        runtime_args.enable_eval_before_training=false \
        models.policy_trainer.num_gpu=${num_device} \
        models.policy_trainer.packing=false \
        models.policy_trainer.bf16=true \
        models.policy_trainer.sequence_parallel=true \
        models.policy_trainer.use_distributed_optimizer=true \
        models.policy_trainer.recompute_granularity='selective' \
        models.policy_trainer.seq_length=2048 \
        models.policy_trainer.tensor_model_parallel_size=4 \
        models.policy_trainer.pipeline_model_parallel_size=2 \
        models.policy_trainer.expert_tensor_parallel_size=4 \
        models.policy_trainer.expert_model_parallel_size=1 \
        models.policy_trainer.generation_batch_size=32 \
        models.policy_trainer.load=${mcore_ckpt_path} \
        models.policy_trainer.optimizer.lr=2e-6 \
        models.policy_trainer.optimizer.min_lr=2e-6 \
        models.policy_trainer.pos_clip_ratio=1e-3 \
        models.policy_trainer.neg_clip_ratio=1e-3 \
        models.policy_trainer.use_group_sequence_policy=true \
        models.reward.generation_batch_size=128 \
        models.policy.load=${hf_ckpt_path} \
        models.policy.generation_batch_size=128 \
        models.policy.tensor_model_parallel_size=4 \
        models.policy.seq_length=2048 \
        models.policy.max_seq_len_to_capture=2348 \
        models.policy.num_inference_per_prompt=32 \
        models.policy.gpu_memory_utilization=0.75 \
        models.policy.enable_thinking=True \
        2>&1 | tee ${log_file} ; exit ${PIPESTATUS[0]}