cd /share/liweihao/repos/Pai-Megatron-Patch-LM4DC/toolkits/model_checkpoints_convertor/deepseek

# hf2mcore
# export MP_PP0_LAYERS=14
# bash hf2mcore_deepseek_v2_moe_convertor.sh \
# A2.4B \
# /share/model/DeepSeek-Coder-V2-Lite-Base \
# /share/liweihao/repos/Pai-Megatron-Patch-LM4DC/examples/deepseek_v2/ckpt-source/DeepSeek-Coder-V2-Lite-Base-to-mcore-tp1-pp2-ep8cls  \
# 1  \
# 2  \
# 8 \
# bf16 \
# false 

# mcore2hf
export MP_PP0_LAYERS=6
bash hf2mcore_deepseek_v2_moe_convertor.sh \
A2.4B \
/share/liweihao/repos/Pai-Megatron-Patch-LM4DC/examples/deepseek_v2/ckpt-train/DeepSeek-Coder-V2-Lite-Base-to-mcore-tp1-pp4-ep2-ghidra-100k/checkpoint/finetune-mcore-deepseek-v2-A2.4B-lr-1e-5-minlr-1e-6-bs-8-gbs-256-seqlen-1024-pr-bf16-tp-1-pp-4-cp-1-ac-full-do-true-sp-true-ti-10-wi-1 \
/share/liweihao/repos/Pai-Megatron-Patch-LM4DC/examples/deepseek_v2/ckpt-convert/DeepSeek-V2-Lite-mcore-te-to-hf    \
1  \
4  \
2 \
bf16 \
true \
/share/model/DeepSeek-Coder-V2-Lite-Base \