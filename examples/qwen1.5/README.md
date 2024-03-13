# QWen 1.5 72B 微调指南
## 模型转换
进入`toolkits/model_checkpoints_convertor/qwen`，将huggingface transformers权重转成megatron
```bash
tp=xxx # 8
pp=xxx # 能够被num_layers整除的数
ori=xxx/Qwen/Qwen1.5-72B-Chat
tgt=xxx/Qwen/Qwen1.5-72B-Chat-hf-to-megatron-tp${tp}-pp${pp}

# https://mp.weixin.qq.com/s?__biz=Mzg4MzgxNDk2OA==&mid=2247491796&idx=1&sn=dc1d719313d794ae1aacdb07669a9545&chksm=cf430783f8348e950218bfcff861a2e6d2d92705807bf5b04f6e9268cc510ffa6e6aa2c87327#rd
bash model_convertor.sh \
    ../../../Megatron-LM-240126 \
    $ori \
    $tgt \
    ${tp} \
    ${pp} \
    qwen1.5-72b \
    0 \
    false

cp ${ori}/merges.txt ${tgt}/merges.txt
```
## 准备测试数据
### SFT数据
```bash
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/alpaca_zh-qwen-train.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/alpaca_zh-qwen-valid.json
```

## 运行微调
每个节点上运行，最后一个node_rank可以查看运行状态
```bash
bash finetune_withGA.sh ${node_rank}
```
finetune_withGA.sh参数释义
- node_rank: 取值范围\[0, n-1\]
- master_addr: 主节点，必须是node_rank 0
- tp/pp: 张量并行/流水线并行，**需要和转换的模型对应，否则会报错**
- seq_len: 序列长度，实测72B在4 * A100 * 8上tp=8/pp=4设置下最长能够达到8192
- extra_vocab_size: 模型embedding_size(可以从报错信息得到，报错的embedding_size * tp) - 千问词表大小(tokenizer.vocab_size返回151643)
