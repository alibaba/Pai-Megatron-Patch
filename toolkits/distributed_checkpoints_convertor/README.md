# [Experimental]分布式模型权重转换

随着模型参数量的增加以及MoE的Expert TP引入，单进程权重转换脚本的问题日益突出。为了解决任意大小模型的Huggingface <-> Megatron-Core的转换问题，

在torch_dist权重格式的基础上，Pai-Megatron-Patch推出一款基于GPU的分布式转换工具。相对于单进程脚本，本工具极大提升了转换速度以及最大支持的参数量。


## 运行须知

0. 当前仅测试gpu转换，如果不使用gpu，backend必须设置为gloo，而非默认的NCCL
1. 建议 TPxPPxEP = n_gpus, ETP=TP 来达到最佳性能
2. 在当前实现中，TP会造成额外读写，建议优先使用PP/EP切分模型，如果需要进一步降低单卡显存占用再开启TP
3. 目前观察到总显存大致为模型大小的1.05x ~ 1.2x, 建议hf2mcore使用1.5x及以上, mcore2hf使用1.8x及以上的总显存以避免OOM
4. mcore2hf转换过程中需要对每轮的通信数据分配内存，内存大小近似正比于saver数量(默认值为gpu数),可以减少saver数量来降低避免OOM
5. 如果遇到NCCL超时问题，欢迎提出issue (对于性能较低的NAS，需要适当延长NCCL超时时间)
6. 训练时，需要将默认的`--ckpt-format torch`设置为`--ckpt-format torch_dist`
7. 本脚本目前不会复制tokenizer，请在转换后自行拷贝需要的配置文件

## 运行方式

根据你需要转换的模型，选择`scripts`文件夹下对应的脚本即可。参数列表如下：

```
MODEL_SIZE=$1               # 模型大小，对于某些模型可能无效
LOAD_DIR=$2                 # 源权重路径
SAVE_DIR=$3                 # 目标权重路径
MG2HF=$4                    # 转换方向 可选: true, false
USE_CUDA=$5                 # 是否使用GPU转换 建议: true
PR=$6                       # 转换精度 可选: fp32 bf16 fp16
HF_DIR=$7                   # HF权重路径(mcore2hf时必须提供)
```

参考运行命令

> 在当前目录 `toolkits/distributed_checkpoints_convertor` 下运行
```
bash scripts/deepseek_v3/run_32xH20.sh \
A37B \
/mnt/deepseek-ckpts/DeepSeek-V3-bf16 \
/mnt/deepseek-ckpts/DeepSeek-V3-to-mcore \
false \
true \
bf16
```

注:

1. Megatron权重路径必须包含`latest_checkpointed_iteration.txt`
2. Huggingface权重路径必须包含`config.json`
3. 当前每个脚本都标注了运行所需的GPU数量，如果需要修改并行配置/使用不同数量GPU拉起，目前可以按照如下方式进行
```
export MODEL_PARALLEL_ARGS="--tensor-model-parallel-size 1 --pipeline-model-parallel-size 8 --expert-tensor-parallel-size 1 --expert-model-parallel-size 4 --decoder-first-pipeline-num-layers 7 --decoder-last-pipeline-num-layers 6"
bash scripts/deepseek_v3/run_32xH20.sh \
A37B \
/mnt/deepseek-ckpts/DeepSeek-V3-bf16 \
/mnt/deepseek-ckpts/DeepSeek-V3-to-mcore \
false \
true \
bf16

```
4. 如果有其他需求，例如机器数与脚本使用的机器数不一致，可以参考`Megatron-LM`及`convert.py`的参数列表自行修改/编写启动脚本。

## 相关性能

我们使用4机32卡转换[DeepSeek-V3-671B](https://github.com/deepseek-ai/DeepSeek-V3),文件系统采用阿里云CPFS。主要时间如下：

> 注: dryrun跳过了模型加载及保存，即移除了几乎所有的io操作

|  Type  |     Model  |     Time      |                                       
|:---------|:--------------:|:--------------:|
| hf2mcore-gpu (dryrun)        |  DeepSeek-V3-671B | 27.3s |
| hf2mcore-gpu        |  DeepSeek-V3-671B | 5min22s |
| mcore2hf-gpu (dryrun)        |  DeepSeek-V3-671B | 47.7s |
| mcore2hf-gpu         |  DeepSeek-V3-671B | 4min43s  |
