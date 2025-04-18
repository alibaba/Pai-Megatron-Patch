#!/bin/bash

# 设置基础变量
CHECKPOINT_DIR="/share/liweihao/repos/Pai-Megatron-Patch-LM4DC/examples/deepseek_v2/ckpt-train/DeepSeek-Coder-V2-Lite-Base-to-mcore-tp1-pp4-ep2-ghidra-100k-fixed/checkpoint/finetune-mcore-deepseek-v2-A2.4B-lr-1e-5-minlr-1e-6-bs-16-gbs-256-seqlen-1024-pr-bf16-tp-1-pp-4-cp-1-ac-full-do-true-sp-true-ti-782-wi-19"
OUTPUT_BASE_DIR="/share/liweihao/repos/Pai-Megatron-Patch-LM4DC/examples/deepseek_v2/ckpt-convert/DeepSeek-Coder-V2-Lite-Base-to-mcore-tp1-pp4-ep2-ghidra-100k-fixed-to-hf"
HF_BASE_MODEL="/share/model/DeepSeek-Coder-V2-Lite-Base"

# 设置环境变量
export MP_PP0_LAYERS=6

# 获取所有iter_开头的目录
for iter_dir in $(ls -rd ${CHECKPOINT_DIR}/iter_*); do
    if [ -d "$iter_dir" ]; then
        # 从目录名中提取迭代号
        iter_num=$(basename $iter_dir | cut -d'_' -f2)
        
        # 创建对应的输出目录名（但先不创建目录）
        output_dir="${OUTPUT_BASE_DIR}/iter_${iter_num}"
        
        # 1. 检查输出目录是否已存在且包含模型文件
        if [ -d "$output_dir" ] && [ -f "$output_dir/model-00004-of-00004.safetensors" ]; then
            echo "Skipping iteration ${iter_num}: Output directory already exists with model file"
            continue
        fi
        
        # 2. 检查源目录是否正在写入
        # 方法1：检查目录的修改时间是否在最近5分钟内
        current_time=$(date +%s)
        dir_modified_time=$(stat -c %Y "$iter_dir")
        time_diff=$((current_time - dir_modified_time))
        
        if [ $time_diff -lt 300 ]; then  # 300秒 = 5分钟
            echo "Skipping iteration ${iter_num}: Source directory was modified in the last 5 minutes, might still be writing"
            continue
        fi
        
        # 方法2：检查是否有.tmp或临时文件
        if ls "$iter_dir"/*.tmp 1> /dev/null 2>&1; then
            echo "Skipping iteration ${iter_num}: Temporary files found, directory might be in use"
            continue
        fi
        
        # 如果通过了所有检查，创建输出目录
        mkdir -p $output_dir
        
        # 更新latest_checkpointed_iteration.txt
        echo $iter_num > "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"
        
        echo "Converting checkpoint from iteration ${iter_num}..."
        
        # 执行转换脚本
        cd /share/liweihao/repos/Pai-Megatron-Patch-LM4DC/toolkits/model_checkpoints_convertor/deepseek

        bash hf2mcore_deepseek_v2_moe_convertor.sh \
            A2.4B \
            "${CHECKPOINT_DIR}" \
            "${output_dir}" \
            1 \
            4 \
            2 \
            bf16 \
            true \
            "${HF_BASE_MODEL}"
        
        echo "Finished converting iteration ${iter_num}"
        echo "----------------------------------------"
    fi
done