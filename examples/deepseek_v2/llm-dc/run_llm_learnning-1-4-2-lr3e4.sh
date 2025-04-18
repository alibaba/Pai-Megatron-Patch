#!/bin/bash

export http_proxy=http://oversea-squid2.ko.txyun:11080 https_proxy=http://oversea-squid2.ko.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

# cd /share/liweihao/repos/Pai-Megatron-Patch/examples/deepseek_v2
cd /share/liweihao/repos/Pai-Megatron-Patch-LM4DC/examples/deepseek_v2

echo "$(realpath ../../)"

export MP_DATASET_TYPE="raw"
export MP_PP0_LAYERS=6


bash -c "
bash run_mcore_deepseek.sh \
dsw  \
A2.4B   \
16   \
256 \
3e-4   \
3e-5   \
2048  \
2048  \
bf16  \
1   \
4  \
1 \
2 \
true \
true   \
true \
true \
full   \
false \
79 \
/share/liweihao/dataset/llm4dc/decompile-ghidra-100k/decompile-ghidra-100k-train.json   \
/share/liweihao/dataset/llm4dc/decompile-ghidra-100k/decompile-ghidra-100k-train.json   \
/share/liweihao/repos/Pai-Megatron-Patch-LM4DC/examples/deepseek_v2/ckpt-source/DeepSeek-Coder-V2-Lite-Base-to-mcore-tp1-pp4-ep2 \
782  \
19   \
/share/liweihao/repos/Pai-Megatron-Patch-LM4DC/examples/deepseek_v2/ckpt-train/DeepSeek-Coder-V2-Lite-Base-to-mcore-tp1-pp4-ep2-ghidra-100k-3e-4
"