#!/bin/bash
# tp1_pp1_expert_0_ep0
# sh hf2mcore_moe.sh /workdir/Pai-megatron /workdir/01ai/Yi-6B /workdir/yi_mg/yi_6b_mcore_tp1_pp1 1 1 0 0 0 false
# tp2_pp1_expert_0_ep0
# sh hf2mcore_moe.sh /workdir/Pai-megatron /workdir/01ai/Yi-6B /workdir/yi_mg/yi_6b_mcore_tp2_pp1 2 1 0 0 0 false
# tp1_pp1_expert_1_ep1
# sh hf2mcore_moe.sh /workdir/Pai-megatron /workdir/01ai/Yi-6B /workdir/yi_mg/yi_6b_mcore_moe1_tp1_pp1_ep1 1 1 0 1 1 false
# tp1_pp1_expert_2_ep2
# sh hf2mcore_moe.sh /workdir/Pai-megatron /workdir/01ai/Yi-6B /workdir/yi_mg/yi_6b_mcore_moe2_tp1_pp1_ep2 1 1 0 2 2 false
# tp2_pp1_expert_2_ep2
# sh hf2mcore_moe.sh /workdir/Pai-megatron /workdir/01ai/Yi-6B /workdir/yi_mg/yi_6b_mcore_moe2_tp2_pp1_ep2 2 1 0 2 2 false
# tp2_pp1_expert_4_ep4
# sh hf2mcore_moe.sh /workdir/Pai-megatron /workdir/01ai/Yi-6B /workdir/yi_mg/yi_6b_mcore_moe4_tp2_pp1_ep4 2 1 0 4 4 false
# tp2_pp1_expert_8_ep4
# sh hf2mcore_moe.sh /workdir/Pai-megatron /workdir/01ai/Yi-6B /workdir/yi_mg/yi_6b_mcore_moe8_tp2_pp1_ep4 2 1 0 8 4 false
# tp2_pp1_expert_8_ep8
# sh hf2mcore_moe.sh /workdir/Pai-megatron /workdir/01ai/Yi-6B /workdir/yi_mg/yi_6b_mcore_moe8_tp2_pp1_ep8 2 1 0 8 8 false
# tp4_pp1_expert_8_ep4
# sh hf2mcore_moe.sh /workdir/Pai-megatron /workdir/01ai/Yi-6B /workdir/yi_mg/yi_6b_mcore_moe8_tp4_pp1_ep4 4 1 0 8 4 false
# tp8_pp1_expert_8_ep2 not support

# tp1_pp1
# sh hf2mcore_moe.sh /workdir/Pai-megatron /workdir/yi_mg/yi_6b_mcore_tp1_pp1 /workdir/01ai_yi/hg_tp1_pp1 1 1 0 0 0 true
# tp2_pp1
# sh hf2mcore_moe.sh /workdir/Pai-megatron /workdir/yi_mg/yi_6b_mcore_tp2_pp1 /workdir/01ai_yi/hg_tp2_pp1 2 1 0 0 0 true
# tp1_pp1_expert_1_ep1
# sh hf2mcore_moe.sh /workdir/Pai-megatron /workdir/yi_mg/yi_6b_mcore_moe1_tp1_pp1_ep1 /workdir/01ai_yi/hg_moe1_tp1_pp1_ep1 1 1 0 1 1 true
# tp1_pp1_expert_2_ep2
# sh hf2mcore_moe.sh /workdir/Pai-megatron /workdir/yi_mg/yi_6b_mcore_moe2_tp1_pp1_ep2 /workdir/01ai_yi/hg_moe2_tp1_pp1_ep2 1 1 0 2 2 true
# tp2_pp1_expert_4_ep4
# sh hf2mcore_moe.sh /workdir/Pai-megatron /workdir/yi_mg/yi_6b_mcore_moe4_tp2_pp1_ep4 /workdir/01ai_yi/hg_moe4_tp2_pp1_ep4 2 1 0 4 4 true
# tp4_pp1_expert_8_ep4
# sh hf2mcore_moe.sh /workdir/Pai-megatron /workdir/yi_mg/yi_6b_mcore_moe8_tp4_pp1_ep4 /workdir/01ai_yi/hg_moe8_tp4_pp1_ep4 4 1 0 8 4 true

set -e
START_TIME=$SECONDS
HG_CKPT_PATH=/workdir/01ai/Yi-6B # ckpt from https://huggingface.co/01-ai/Yi-6B
export CUDA_VISIBLE_DEVICES=5
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
MEGATRON_PATH=$1
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATH}:${MEGATRON_PATH}/Megatron-LM-main
SOURCE_CKPT_PATH=$2
TARGET_CKPT_PATH=$3
TP=$4
PP=$5
EXTRA_VOCAB_SIZE=$6
NUM_EXPERTS=$7 # target expert num
EP=$8
mg2hf=$9


if [ $NUM_EXPERTS -gt 0 ]; then
    expert_options="
                --moe-router-topk 1 \
                --num-experts ${NUM_EXPERTS} \
                --expert-model-parallel-size 1 \
                --target_expert_model_parallel_size ${EP}
    "
fi

if [ $mg2hf = true ]; then
    do_options="
                --convert_checkpoint_from_megatron_to_transformers
    "
elif [ $mg2hf = false ]; then
    do_options=""
fi

DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

torchrun ${DISTRIBUTED_ARGS} hf2mcore_moe.py \
    --load_path ${SOURCE_CKPT_PATH} \
    --save_path ${TARGET_CKPT_PATH} \
    --huggingface_model_path ${HG_CKPT_PATH} \
    --megatron-path ${MEGATRON_PATH} \
    --target_tensor_model_parallel_size ${TP} \
    --target_pipeline_model_parallel_size ${PP} \
    --use-cpu-initialization \
    --bf16 \
    --num-layers 32 \
    --hidden-size 4096 \
    --micro-batch-size 1 \
    --ffn-hidden-size 11008 \
    --num-attention-heads 32 \
    --num-query-groups 4 \
    --hidden-dropout 0.0 \
    --attention-dropout 0.0 \
    --normalization RMSNorm \
    --max-position-embeddings 4096 \
    --encoder-seq-length 4096 \
    --patch-tokenizer-type YiTokenizer \
    --no-async-tensor-model-parallel-allreduce \
    --load ${SOURCE_CKPT_PATH} \
    --swiglu \
    --disable-bias-linear \
    --no-rope-fusion \
    --group-query-attention \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --rotary-base 5000000 \
    --vocab-size 64000 \
    --use-rotary-position-embeddings \
    --transformer-impl transformer_engine \
    --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
    --untie-embeddings-and-output-weights \
    ${expert_options}  ${do_options} 

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"