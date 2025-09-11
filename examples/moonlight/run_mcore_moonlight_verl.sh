#!/bin/bash

ray stop
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
VERL_PATCH_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=${VERL_PATCH_PATH}:${VERL_PATCH_PATH}/backends/megatron/Megatron-LM-250624:${VERL_PATCH_PATH}/backends/rl/verl:$PYTHONPATH
export HYDRA_FULL_ERROR=1
gsm8k_train_path=/mnt/data/datasets/gsm8k/train.parquet
gsm8k_test_path=/mnt/data/datasets/gsm8k/test.parquet

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

HF_MODEL_PATH=/mnt/data/ckpts/huggingface/Moonlight-16B-A3B-Instruct
DIST_CKPT_PATH=/mnt/data/ckpts/mcore/Moonlight-16B-A3B-Instruct-to-mcore


# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping

export GPUS_PER_NODE=${MLP_WORKER_GPU:-${KUBERNETES_CONTAINER_RESOURCE_GPU:-8}}
export RAY_num_server_call_thread=1
export NNODES=${MLP_WORKER_NUM:-${WORLD_SIZE:-1}}
export NODE_RANK=${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-0}}}
export MASTER_ADDR=${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}
export MASTER_PORT=${MLP_WORKER_0_PORT:-${MASTER_PORT:-1234}}

set -x
if [ $NODE_RANK -eq 0 ]; then

ray start --block --head --port=6379 &

python verl_entrypoint.py --config-path=verl_configs \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    +actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=32 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_grpo_example_gsm8k_math' \
    trainer.experiment_name='qwen3_30b_moe_megatron' \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.total_epochs=50 2>&1 | tee patch_verl_demo_megatron.log
else
ray start --block --address=${MASTER_ADDR}:6379
fi