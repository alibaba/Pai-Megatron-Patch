## 简介
Pai-Megatron-Patch工具是阿里云机器学习平台PAI算法团队研发，基于灵骏智算平台的大模型最佳实践解决方案配套工具，旨在帮助大模型开发者快速上手灵骏产品，完成大语言模型（LLM）的高效分布式训练，有监督指令微调，模型离线推理验证等完整大模型开发链路。该项目提供了基于Megatron-LM的大模型训练&离线推理验证流程，方便用户快速上手大模型训练。

Pai-Megatron-Patch工具基于英伟达的Megatron-LM框架来构建大模型开发链路。设计上对Megatron-LM源码非侵入式修改，采用补丁方式对接Megatron-LM训练加速库。patch中包含模型库，分词器，模型转换，强化学习，离线文本生成以及使用示例和工具集。在模型库中包含热门大模型的Megatron版本实现以及对应的数据ID化流程。同时patch提供了huggingface模型权重到megatron模型权重的转换，方便用户加载huggingface的权重进行继续预训练或者微调。patch还提供了对训练好的megatron模型权重到huggingface模型权重的转换，方便用户使用huggingface的评估流程对模型质量进行客观评估或者使用huggingface的离线文本生成流水线。在强化学习部分，patch提供了PPO训练流程等，方便用户使用SFT模型和RM模型进行强化学习。同时patch提供了大量的使用示例帮助用户快速开始大模型训练&离线推理。目前patch中支持的大模型有主要baichuan，bloom，chatglm，falcon，galactica，glm，llama，qwen和starcoder，后续还会根据需要及时添加新的Megatron版大模型实现。
具体在阿里云灵骏产品的使用流程请参考: [智算服务PAI灵骏大模型分布式训练方案](https://help.aliyun.com/document_detail/2505831.html?spm=5176.28352543.J_9l_YP1wy4J7aEdtojTyUD.1.347850adeLHhmP&tab=onestop)

<div align=center>
<img src=patch.png width=600 height=400 />
</div>

## 快速开始

[环境准备](https://help.aliyun.com/document_detail/2505831.html?spm=5176.28352543.J_9l_YP1wy4J7aEdtojTyUD.1.347850adeLHhmP&tab=onestop)

[数据准备](toolkits/pretrain_data_preprocessing/README.md)

[HFDS有监督微调&继续预训练](examples/hfds.md)

[Megatron有监督微调&继续预训练](examples/megatron.md)

[人类反馈强化学习](rlhf/README.md)

[模型离线推理](megatron_patch/generation/megatron.md)


## License
This project is licensed under the [Apache License (Version 2.0)](https://github.com/alibaba/pai-megatron-patch/blob/master/LICENSE). This toolkit also contains some code modified from other repos under other open-source licenses. See the [NOTICE](https://github.com/alibaba/pai-megatron-patch/blob/master/NOTICE) file for more information.
