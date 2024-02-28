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
- [基于Megatron-Core的稀疏大模型训练工具：阿里云MoE大模型最佳实践](https://mp.weixin.qq.com/s/DkrWEEJ7IxirwWd3qB9Bng)
- [Mixtral-8x7B在PAI灵骏的训练指南](https://help.aliyun.com/zh/pai/use-cases/train-fine-tune-and-deploy-mixtral-by-using-intelligent-computing-lingjun)
- [通义千问开源模型在PAI灵骏的最佳实践](https://mp.weixin.qq.com/s?__biz=Mzg4MzgxNDk2OA==&mid=2247491796&idx=1&sn=dc1d719313d794ae1aacdb07669a9545&chksm=cf430783f8348e950218bfcff861a2e6d2d92705807bf5b04f6e9268cc510ffa6e6aa2c87327#rd)
- [阿里云机器学习PAI开源AI大模型训练工具Pai-Megatron-Patch, 助力大模型技术落地](https://zhuanlan.zhihu.com/p/655942437)
- [基于单机最高能效270亿参数GPT模型的文本生成与理解](https://zhuanlan.zhihu.com/p/597652820)
- [中文稀疏GPT大模型落地 — 通往低成本&高性能多任务通用自然语言理解的关键里程碑](https://zhuanlan.zhihu.com/p/561320982)
- [预训练知识度量比赛夺冠！阿里云PAI发布知识预训练工具](https://zhuanlan.zhihu.com/p/449487792)
- [阿里云PAI获得FewCLUE基于大模型的小样本学习双料冠军](https://developer.aliyun.com/article/788081?spm=a2c6h.12873639.article-detail.17.11c5383cHpFZks&tlog=yuekan_8)

新功能：
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

## 安装

```bash
$ git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
```

## 快速开始

[环境准备](https://help.aliyun.com/document_detail/2505831.html?spm=5176.28352543.J_9l_YP1wy4J7aEdtojTyUD.1.347850adeLHhmP&tab=onestop)

[数据准备](toolkits/pretrain_data_preprocessing/README.md)

[HFDS有监督微调&继续预训练](examples/hfds.md)

[Megatron有监督微调&继续预训练](examples/megatron.md)

[人类反馈强化学习](rlhf/README.md)

[模型离线推理](megatron_patch/generation/megatron.md)



## 技术交流群
欢迎使用[钉钉](https://www.dingtalk.com/en)扫描如下的二维码进群交流
<div align=center>
<img src=qr.png width=600 height=450 />
</div>

## 许可证
本项目采用 [Apache License (Version 2.0)](https://github.com/alibaba/pai-megatron-patch/blob/master/LICENSE).
本项目包含来自于其他项目的开源许可授权的代码，具体请查看[NOTICE](https://github.com/alibaba/pai-megatron-patch/blob/master/NOTICE).
