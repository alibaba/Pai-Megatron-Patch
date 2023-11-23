## Introduction
English | [简体中文](./README_zh-CN.md)

Pai-Megatron-Patch (https://github.com/alibaba/Pai-Megatron-Patch) is a deep learning training toolkit built for developers to train and predict large language models (LLMs) by using MegatronLM framework easily. With the continuous development of LLMs, the model structure and scale are rapidly evolving. Although these models can be conveniently manufactured using Transformers or DeepSpeed training framework, the training efficiency is comparably low. This phenomenon becomes even severer when the model scale exceeds 10 billion. The primary objective of Pai-Megatron-Patch is to effectively utilize the computational power of GPUs for LLM. This tool allows convenient training of commonly used LLM with all the accelerating techniques provided by Megatron-LM.

## Highlights
Pai-Megatron-Patch is developed by the Alibaba Cloud Machine Learning Platform (PAI) algorithm team.  The tool aims to assist developers in quickly getting started with Lingjun products and completing the entire development pipeline for LLM, including efficient distributed training, supervised fine-tuning, and offline model inference or verification. It has several merits as follows:

- Support for multiple commonly used LLM such as llama, llama-2, codellama, baichuan, qwen, Falcon, GLM, Starcoder, Bloom, chatglm, etc.
- Support for model weight conversion: Mapping operator namespaces between Huggingface, Megatron, and Transformer Engine.
- Support for FP8 training acceleration in Flash Attention 2.0 and Transformer Engine modes, ensuring training convergence.
- Rich and user-friendly usage examples, offering best practices for the entire workflow of LLM pre-training, fine-tuning, evaluation, and inference, as well as reinforcement learning.
## Framework
The design philosophy of Pai-Megatron-Patch is to avoid invasive modifications to the source code of Megatron-LM. In other words, it does not add new modules directly to Megatron-LM. Instead, the functions that need expansion and improvement are presented in the form of patch. This decoupling ensures that users can continue to embrace the best practices of LLM without being affected by upgrades of Megatron-LM.

Pai-Megatron-Patch includes key components for building LLM training, such as model library, tokenizers, model convertors, reinforcement learning , offline text generation, usages examples, and toolkits. The model library provides popular LLMs implemented in Megatron, such as baichuan, bloom, chatglm, falcon, galactica, glm, llama, qwen, and starcoder. More Megatron-based implementations of LLMs will be added as needed in the future. Additionally, the patch provides bidirectional conversion between Huggingface and Megatron model weights. This allows users to easily utilize Huggingface pretrained models for continued pre-training or fine-tuning in Megatron, as well as evaluating model quality using Huggingface's evaluation/inference pipelines on trained Megatron models.

In the reinforcement learning section, the patch offers PPO training workflows, enabling users to perform reinforcement learning with SFT models and RM models. Finally, the patch provides numerous usage examples to help users quickly start LLMs training and offline inference. For specific usage processes within Alibaba Cloud Lingjun products, please refer to the following link: [PAI-Lingjun Intelligent Computing Service LLM solution](https://www.aliyun.com/solution/tech-solution/pai_lingjun).

<div align=center>
<img src=patch_en.png width=600 height=400 />
</div>

## Installation

```bash
$ git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
```

## Technical Reports

- [阿里云机器学习PAI开源AI大模型训练工具Pai-Megatron-Patch, 助力大模型技术落地](https://zhuanlan.zhihu.com/p/655942437)
- [基于单机最高能效270亿参数GPT模型的文本生成与理解](https://zhuanlan.zhihu.com/p/597652820)
- [中文稀疏GPT大模型落地 — 通往低成本&高性能多任务通用自然语言理解的关键里程碑](https://zhuanlan.zhihu.com/p/561320982)
- [预训练知识度量比赛夺冠！阿里云PAI发布知识预训练工具](https://zhuanlan.zhihu.com/p/449487792)
- [阿里云PAI获得FewCLUE基于大模型的小样本学习双料冠军](https://developer.aliyun.com/article/788081?spm=a2c6h.12873639.article-detail.17.11c5383cHpFZks&tlog=yuekan_8)


## Quick Start

[Environment Preparation](https://help.aliyun.com/document_detail/2505831.html?spm=5176.28352543.J_9l_YP1wy4J7aEdtojTyUD.1.347850adeLHhmP&tab=onestop)

[Data Preparation](toolkits/pretrain_data_preprocessing/README.md)

[Huggingface Training](examples/hfds.md)

[Megatron Training](examples/megatron.md)

[RLHF](rlhf/README.md)

[Inference](megatron_patch/generation/megatron.md)

## Contact
Use [Dingtalk](https://www.dingtalk.com/en) to scan blow QR code
<div align=center>
<img src=qr.png width=600 height=450 />
</div>

## License
This project is licensed under the [Apache License (Version 2.0)](https://github.com/alibaba/pai-megatron-patch/blob/master/LICENSE). This toolkit also contains some code modified from other repos under other open-source licenses. See the [NOTICE](https://github.com/alibaba/pai-megatron-patch/blob/master/NOTICE) file for more information.
