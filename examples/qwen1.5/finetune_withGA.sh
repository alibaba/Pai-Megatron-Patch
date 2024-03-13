# 手动修改
export PYTHONPATH=xxx
# master节点必须是node rank 0
master_addr=xxx
nnodes=xxx

tp=8
pp=4
model_size=72B
extra_vocab_size=421

global_batch_size=32
lr=2e-5
min_lr=5e-6
seq_len=xxx
pad_len=$seq_len
run_name=qwen1-5_${model_size}_megatron_test

dataset_path=xxx/qwen-datasets/alpaca_zh-qwen-train.json
valid_dataset_path=xxx/qwen-datasets/alpaca_zh-qwen-valid.json
pretrain_checkpoint_path=xxx/Qwen/Qwen1.5-${model_size}-Chat-hf-to-megatron-tp${tp}-pp${pp}
data_length=$(jq 'length' $dataset_path)
epoch=2
train_iters=$(( (data_length / global_batch_size) * epoch ))

save_interval=0.5
save_steps=$(echo "$train_iters * $save_interval" | bc | awk '{print int($1+0.5)}')
warmup_ratio=0.01
lr_warmup_iters=$(echo "$train_iters * $warmup_ratio" | bc | awk '{print int($1+0.5)}')
output_basepath=xxx/saved_models/${run_name}_${epoch}

echo "data_length: $data_length"
echo "epoch: $epoch"
echo "global_batch_size: $global_batch_size"
echo "train_iters: $train_iters"
echo "save_steps: $save_steps"
echo "lr_warmup_iters: $lr_warmup_iters"
echo "output_basepath: $output_basepath"

export WORLD_SIZE=$nnodes
export RANK=$1
export KUBERNETES_CONTAINER_RESOURCE_GPU=8
export MASTER_ADDR=$master_addr
export MASTER_PORT=30000

# 根据自己的集群情况修改
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_TC=136
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=5
export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_NET_PLUGIN=none
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9

# https://mp.weixin.qq.com/s?__biz=Mzg4MzgxNDk2OA==&mid=2247491796&idx=1&sn=dc1d719313d794ae1aacdb07669a9545&chksm=cf430783f8348e950218bfcff861a2e6d2d92705807bf5b04f6e9268cc510ffa6e6aa2c87327#rd
bash run_finetune_megatron_qwen_withGA.sh \
    dlc \
    ../../Megatron-LM-240126 \
    $model_size \
    1 \
    $global_batch_size \
    $lr \
    $min_lr \
    $seq_len \
    $pad_len \
    $extra_vocab_size \
    bf16 \
    $tp \
    $pp \
    sel \
    true \
    true \
    true \
    false \
    $save_steps \
    $dataset_path \
    $valid_dataset_path \
    $pretrain_checkpoint_path \
    $train_iters \
    $lr_warmup_iters \
    $output_basepath