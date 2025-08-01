## 快速开始

|             |                                                      Megatron-Core                                                       |                                                                                        ChatLearn                                                                                        |    verl     |
|:------------|:------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------:|
| Qwen3       |      [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/qwen3/README.md#Megatron-Core模型训练流程)      |                                             [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/qwen3/README_grpo.md)                                             | Coming Soon |
| QwQ         |       [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/qwq/README.md#Megatron-Core模型训练流程)       |                                                                                           N/A                                                                                           |     N/A     |
| Qwen2.5-VL  |   [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/qwen2_5_vl/README.md#Megatron-Core模型训练流程)    |                                                                                           N/A                                                                                           |     N/A     |
| Moonlight   |  [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/moonlight/README.md#Megatron-Core-MoE模型训练流程)  |                                           [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/moonlight/README_grpo.md)                                           | Coming Soon |
| DeepSeek-V3 | [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/deepseek_v3/README.md#Megatron-Core-MoE模型训练流程) |                                         [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/deepseek_v3/README_grpo.md)                                                                                     | Coming Soon |
| Qwen2-VL    |    [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/qwen2_vl/README.md#Megatron-Core模型训练流程)     |                                                                                           N/A                                                                                           |     N/A     |
| LLaVA       |   [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/llava_mcore/README.md#Megatron-Core模型训练流程)   |                                                                                           N/A                                                                                           |     N/A     |
| Qwen2.5     |  [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/qwen2_5/README.md#Megatron-Core-Dense模型训练流程)  |                                                                                           N/A                                                                                           |     N/A     |  
| LLama3.1    | [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/llama3_1/README.md#Megatron-Core-Dense模型训练流程)  |                                                                                           N/A                                                                                           |     N/A     |
| LLama3      |  [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/llama3/README.md#Megatron-Core-Dense模型训练流程)   |                                                                                           N/A                                                                                           |     N/A     |
| LLama2      |  [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/llama2/README.md#Megatron-Core-Dense模型训练流程)   |                                                                                           N/A                                                                                           |     N/A     |
| Mistral     |     [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/mistral/README.md#Megatron-Core模型训练流程)     |                                                                                           N/A                                                                                           |     N/A     |
| Qwen2       |      [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/qwen2/README.md#Megatron-Core模型训练流程)      |                                                                                           N/A                                                                                           |     N/A     |
| Qwen1.5     |  [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/qwen1_5/README.md#Megatron-Core-Dense模型训练流程)  |                                                                                           N/A                                                                                           |     N/A     |
| DeepSeek-V2 | [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/deepseek_v2/README.md#Megatron-Core-MoE模型训练流程) |                                                                                           N/A                                                                                           |     N/A     |

## Pai-Megatron-Patch是什么
[English](./README.md) | 简体中文

随着深度学习大模型的不断发展，其模型结构和量级在快速演化，依托大模型技术的应用更是层出不穷。
对于广大开发者来说不仅要考虑如何在复杂多变的场景下有效的将大模型消耗的算力发挥出来，还要应对大模型的持续迭代。
开发简单易用的大模型训练工具就成了应对以上问题广受关注的技术方向，让开发者专注于大模型解决方案的开发，降低大模型训练加速性能优化和训练/推理全流程搭建的人力开发成本。

Pai-Megatron-Patch工具是阿里人工智能平台PAI算法团队研发，基于阿里云智算服务PAI-灵骏平台的大模型最佳实践解决方案配套工具。
Pai-Megatron-Patch是各类开源大模型和Megatron训练加速引擎之间的“桥梁”，为用户提供用Megatron训练开源大模型的易用性以及LLM算法场景定制化的灵活性。
同时它可以帮助大模型开发者快速上手PAI灵骏产品，完成大模型的高效分布式训练，有监督指令微调，模型离线推理验证等完整大模型开发链路。
该项目提供了业界主流开源大模型基于Megatron的训练&离线推理验证流程，方便用户快速上手大模型训练。


## Technical Reports
- [基于 Megatron 的多模态大模型训练加速技术解析](https://mp.weixin.qq.com/s/USMOkRuy-J5UpxyluqsBmg)
- [Pai-Megatron-Patch：围绕Megatron-Core打造大模型训练加速生态](https://mp.weixin.qq.com/s/BGyiJ90ZB75s3EP74KogeA)
- [Meta Llama3.1模型在PAI-Megatron-Patch的最佳实践](https://help.aliyun.com/zh/pai/use-cases/best-practice-for-llama-3-1-in-pai-megatron-patch?spm=a2c4g.11186623.0.0.4cef730eqWHNY7)
- [基于Megatron-Core的稀疏大模型训练工具：阿里云MoE大模型最佳实践](https://mp.weixin.qq.com/s/DkrWEEJ7IxirwWd3qB9Bng)
- [Mixtral-8x7B在PAI灵骏的训练指南](https://help.aliyun.com/zh/pai/use-cases/train-fine-tune-and-deploy-mixtral-by-using-intelligent-computing-lingjun)
- [通义千问开源模型在PAI灵骏的最佳实践](https://mp.weixin.qq.com/s?__biz=Mzg4MzgxNDk2OA==&mid=2247491796&idx=1&sn=dc1d719313d794ae1aacdb07669a9545&chksm=cf430783f8348e950218bfcff861a2e6d2d92705807bf5b04f6e9268cc510ffa6e6aa2c87327#rd)
- [阿里云机器学习PAI开源AI大模型训练工具Pai-Megatron-Patch, 助力大模型技术落地](https://zhuanlan.zhihu.com/p/655942437)
- [基于单机最高能效270亿参数GPT模型的文本生成与理解](https://zhuanlan.zhihu.com/p/597652820)
- [中文稀疏GPT大模型落地 — 通往低成本&高性能多任务通用自然语言理解的关键里程碑](https://zhuanlan.zhihu.com/p/561320982)
- [预训练知识度量比赛夺冠！阿里云PAI发布知识预训练工具](https://zhuanlan.zhihu.com/p/449487792)
- [阿里云PAI获得FewCLUE基于大模型的小样本学习双料冠军](https://developer.aliyun.com/article/788081?spm=a2c6h.12873639.article-detail.17.11c5383cHpFZks&tlog=yuekan_8)

新功能：
- **支持DeepSeek-V3-671B模型使用Mcore+ChatLearn进行强化学习GRPO训练** [🔥🔥 2025.07.31]
- **支持Qwen3-235B模型使用Mcore+ChatLearn进行强化学习GRPO训练** [🔥🔥 2025.07.20]
- **支持Moonlight模型使用Mcore+ChatLearn进行强化学习GRPO训练** [🔥🔥 2025.06.30]
- **支持Qwen3模型使用Mcore+ChatLearn进行强化学习GRPO训练** [🔥🔥 2025.06.03]
- **支持Qwen2.5模型使用Mcore+ChatLearn进行强化学习GRPO训练** [🔥🔥 2025.05.18]
- **支持全系列Qwen3模型基于torch_dist权重格式的训练微调** [🔥🔥 2025.04.29]
- **[实验性]实现用于超大参数量模型的MG/HF权重分布式转换** [🔥🔥 2025.04.16]
- **升级完善DeepSeek-V3训练微调流程** [🔥🔥 2025.03.31]
- **支持用Megatron-Core框架训练QwQ模型** [🔥🔥 2025.03.27]
- **支持用Megatron-Core框架训练Qwen2.5-VL模型** [🔥🔥 2025.03.21]
- **支持用Megatron-Core框架训练来自月之暗面KIMI的Moonlight-16B-A3B模型** [🔥🔥 2025.03.14]
- **优化DeepSeek-V3模型转换脚本，支持DeepSeek-V3模型的专家并行转换** [🔥🔥 2025.03.14]
- **支持用Megatron-Core框架训练DeepSeek-V3模型** [🔥🔥 2025.02.21]
- **升级SFT微调流程** [🔥🔥 2025.02.20]
- **升级DeepSeek-V2-MoE模型最佳实践为接入DeepSeek-V3-MoE的工程加速过渡** [🔥🔥 2025.01.16]
- **拓展Qwen2-VL模型以支持序列并行、虚拟流水并行及TP-Comm-Overlap特性** [🔥🔥 2025.01.15]
- **拓展Qwen2-VL模型权重转换及多轮复杂多模态数据的训练支持** [🔥🔥 2024.12.27]
- **支持用Megatron-Core框架训练Qwen2-VL模型** [🔥🔥 2024.11.27]
- **支持用Megatron-Core框架训练LLaVA模型** [🔥🔥 2024.11.20]
- **添加大模型训练最优吞吐参数自动配置以及针对qwen2/2.5系列模型优化微调per seq sft loss.** [🔥🔥 2024.10.30]
- **升级Deepseek-V2-MoE系列模型支持TE版的MLA以及流水并行CKPT转换** [🔥🔥 2024.09.26]
- **支持用Megatron-Core框架训练Qwen2.5系列模型** [🔥🔥 2024.09.20]
- **支持Qwen2及LLaMA-3.1系列模型SFT的Sequence Packing技术.** [🔥🔥 2024.09.13]
- **升级Qwen2系列模使用Flash-Attention 3, Offloading, Comm-Overlapping训练微调** [🔥🔥 2024.08.23]
- **支持LLaMA-3.1系列模使用Flash-Attention3训练微调** [🔥🔥 2024.08.23]
- **支持用Megatron-Core框架训练LLaMA-3.1系列模型** [🔥🔥 2024.08.23]
- **支持自动优化器卸载.** [🔥🔥 2024.07.25]
- **支持静态优化器卸载.** [🔥🔥 2024.07.15]
- **支持用Megatron-Core框架训练qwen-2-MoE系列模型** [🔥🔥 2024.06.19]
- **支持用Megatron-Core框架训练qwen-2-Dense系列模型** [🔥🔥 2024.06.12]
- **支持用Megatron-Core框架训练deepseek-v2-MoE系列模型** [🔥🔥 2024.05.30]
- **支持用Megatron-Core框架训练qwen1.5-MoE系列模型** [🔥🔥 2024.05.13]
- **支持用Megatron-LM和Megatron-Core框架训练llama3系列模型** [🔥🔥 2024.04.21]
- **支持用Megatron-Core框架训练qwen1.5系列模型** [🔥🔥 2024.03.20]
- **支持用Megatron-LM框架训练qwen1.5系列模型** [🔥🔥 2024.02.28]
- **支持用Megatron-Core框架训练mixtral-8x7b MoE稀疏模型** [🔥🔥 2024.01.26]
- **支持用Megatron-LM框架训练多模态大模型qwen-vl.** [🔥🔥 2023.12.15]
- **支持用Megatron-LM框架训练多模态大模型LLava.** [🔥🔥 2023.12.01]
- **支持用Megatron-LM框架训练deepseek系列模型.** [🔥🔥 2023.11.24]
- **支持用Megatron-LM框架训练qwen-72B模型.** [🔥🔥 2023.11.23]
- **支持用Megatron-LM框架训练Mistral-7B, Yi-6B和Codellama-34B模型** [🔥🔥 2023.11.16]
- **升级Megatron-LM底座，帮助热门模型支持transformer engine和fp8训练.** [🔥🔥 2023.10.19]
- **支持用Megatron-LM框架训练qwen-14B和baichuan2-13B** [🔥🔥 2023.10.08]

## 主要特性

* 多款热门大模型支持：llama，llama-2系列，codellama， deepseek，百川，通义千问，Falcon，GLM，Starcoder，Bloom，chatglm等
* 支持模型权重互转转换：在Huggingface，Megatron和Transformer Engine之间进行算子命名空间映射
* 支持Flash Attention 2.0和Transformer Engine模式下的FP8训练加速且确保收敛
* 丰富且简单易用的使用示例，支持大模型预训练，微调，评估和推理，强化学习全流程最佳实践

## 技术架构

Pai-Megatron-Patch的设计理念是不对Megatron-LM的源码进行侵入式修改，即不在Megatron-LM里面添加新的功能特性，
将需要扩充完善的部分以patch补丁的方式呈现。在patch中构建LLM训练链路通过依赖Megatron-LM核心库的方法实现和Megatron-LM的解耦合。
这样解耦合的好处就是Megatron-LM的升级不会影响用户的LLM最佳实践体验。

Pai-Megatron-Patch中包含模型库，分词器，模型转换，强化学习，离线文本生成以及使用示例和工具集等用于构建LLM训练的关键要素。
在模型库中包含热门大模型的Megatron版本实现，例如baichuan，bloom，chatglm，falcon，galactica，glm，llama，qwen和starcoder，
后续还会根据需要及时添加新的Megatron版大模型实现。同时patch还提供了huggingface模型权重和Megatron模型权重之间的双向转换。
一方面是方便用户加载huggingface的权重在Megatron中继续预训练或者微调，
另一方面是方便用户对训练好的Megatron模型使用huggingface的评估/推理流程对模型质量进行客观评估。
在强化学习部分，patch提供了PPO训练流程等，方便用户使用SFT模型和RM模型进行强化学习。最后patch提供了大量的使用示例帮助用户快速开始大模型训练&离线推理。

具体在阿里云灵骏产品的使用流程请参考: [智算服务PAI灵骏大模型分布式训练方案](https://www.aliyun.com/solution/tech-solution/pai_lingjun)


<div align=center>
<img src=patch.png width=600 height=400 />
</div>


## 技术交流群
欢迎使用[钉钉](https://www.dingtalk.com/en)扫描如下的二维码进群交流, 1和2群已满，请加3群。
<div align=center>
<img src=qr.png width=600 height=450 />
</div>

<div align=center>
<img src=qr2.png width=600 height=450 />
</div>

<div align=center>
<img src=qr3.png width=600 height=450 />
</div>

## 许可证
本项目采用 [Apache License (Version 2.0)](https://github.com/alibaba/pai-megatron-patch/blob/master/LICENSE).
本项目包含来自于其他项目的开源许可授权的代码，具体请查看[NOTICE](https://github.com/alibaba/pai-megatron-patch/blob/master/NOTICE).
