#!/bin/bash
set -x

ray stop
rm -rf /tmp/ray/*

export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=${MLP_WORKER_GPU:-${KUBERNETES_CONTAINER_RESOURCE_GPU:-8}}
export RAY_num_server_call_thread=1
export NNODES=${MLP_WORKER_NUM:-${WORLD_SIZE:-1}}
export NODE_RANK=${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-0}}}
export MASTER_ADDR=${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}
export MASTER_PORT=${MLP_WORKER_0_PORT:-${MASTER_PORT:-1234}}

CURRENT_DIR=$(pwd)
MEGATRON_PATCH_PATH=$( dirname $( dirname ${CURRENT_DIR}))
VERL_ROOT_PATH=${MEGATRON_PATCH_PATH}/backends/rl/verl
export PYTHONPATH=${MEGATRON_PATCH_PATH}:${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-250624:${VERL_ROOT_PATH}:$PYTHONPATH

export RAY_CGRAPH_get_timeout=200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_num_server_call_thread=1
export RAY_DEDUP_LOGS=0
export VLLM_USE_RAY_SPMD_WORKER=1
export VLLM_USE_RAY_COMPILED_DAG=1

train_path=/mnt/data/datasets/MATH-lighteval/train.parquet
test_path=/mnt/data/datasets/MATH-lighteval/test.parquet

train_files="['$train_path']"
test_files="['$test_path']"

hf_ckpt_path=/mnt/data/ckpts/huggingface/Moonlight-16B-A3B-Instruct
mcore_ckpt_path=/mnt/data/ckpts/mcore/Moonlight-16B-A3B-Instruct-to-mcore
proj_name="jerry_debug"
exp_name="test_moonlight_16b"
export output_dir=${CURRENT_DIR}/verl_outputs/${exp_name}
export WANDB_DIR=${output_dir}
mkdir -p $output_dir/
export log_dir=${output_dir}/logs
mkdir -p $log_dir
log_file=$log_dir/${exp_name}_rank${NODE_RANK}.log

if [ $NODE_RANK -eq 0 ]; then

ray start --block --head --port=6379 &

python ../qwen3/verl_entrypoint.py --config-path=../qwen3/verl_configs \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=8 \
    data.max_prompt_length=1536 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path=$hf_ckpt_path \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    +actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$mcore_ckpt_path \
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
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=$mcore_ckpt_path \
    actor_rollout_ref.model.trust_remote_code=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=${proj_name} \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=200 \
    2>&1 | tee ${log_file} ; exit ${PIPESTATUS[0]}
else
ray start --block --address=${MASTER_ADDR}:6379
fi