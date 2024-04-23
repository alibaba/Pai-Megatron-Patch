## Quick Start


|         |                                                  Megatron-LM-Dense                                                   |                                                  Megatron-Core-Dense                                                   |                                                  Megatron-Core-MoE                                                   | MegaBlocks-MoE |
|:--------|:--------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------:|:--------------:|
| LLama3  | [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/llama3/README.md#Megatron-LM-Dense模型训练流程)  | [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/llama3/README.md#Megatron-Core-Dense模型训练流程)  |                                                         N/A                                                          |      N/A       |
| LLama2  | [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/llama2/README.md#Megatron-LM-Dense模型训练流程)  | [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/llama2/README.md#Megatron-Core-Dense模型训练流程)  |                                                         N/A                                                          |      N/A       |
| Mistral | [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/mistral/README.md#Megatron-LM-Dense模型训练流程) | [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/mistral/README.md#Megatron-Core-Dense模型训练流程) | [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/mistral/README.md#Megatron-Core-MoE模型训练流程) |      N/A       |
| Qwen1.5 | [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/qwen1_5/README.md#Megatron-LM-Dense模型训练流程) | [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/qwen1_5/README.md#Megatron-Core-Dense模型训练流程) | [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/qwen1_5/README.md#Megatron-Core-MoE模型训练流程) |      [ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/qwen1_5/README.md#MegaBlocks-MoE模型训练流程)        |


## Introduction
English | [简体中文](./README_zh-CN.md)

Pai-Megatron-Patch (https://github.com/alibaba/Pai-Megatron-Patch) is a deep learning training toolkit built for developers to train and predict LLMs & VLMs by using Megatron framework easily. With the continuous development of LLMs, the model structure and scale are rapidly evolving. Although these models can be conveniently manufactured using Transformers or DeepSpeed training framework, the training efficiency is comparably low. This phenomenon becomes even severer when the model scale exceeds 10 billion. The primary objective of Pai-Megatron-Patch is to effectively utilize the computational power of GPUs for LLM. This tool allows convenient training of commonly used LLM with all the accelerating techniques provided by Megatron-LM.

What's New:
- **Support training llama3 models by using Megatron-LM and Megatron-Core.** [🔥🔥 2024.04.21]
- **Support training qwen1.5 models by using Megatron-Core.** [🔥🔥 2024.03.20]
- **Support training qwen1.5 models by using Megatron-LM.** [🔥🔥 2024.02.28]
- **Support training mixtral-8x7b moe model by using Megatron-Core.** [🔥🔥 2024.01.26]
- **Support training qwen-vl multimodel by using Megatron-LM.** [🔥🔥 2023.12.15]
- **Support training LLava multimodel by using Megatron-LM.** [🔥🔥 2023.12.01]
- **Support training deepseek model by using Megatron-LM.** [🔥🔥 2023.11.24]
- **Support training qwen-72B model by using Megatron-LM.** [🔥🔥 2023.11.23]
- **Support training Mistral-7B, Yi-6B and Codellama-34B** [🔥🔥 2023.11.16]
- **Upgrade Megatron-LM for Llama2, qwen and baichuan2 to use transformer engine and fp8.** [🔥🔥 2023.10.19]
- **Support training qwen-14B and baichuan2-13B model by using Megatron-LM.** [🔥🔥 2023.10.08]

## Highlights
Pai-Megatron-Patch is developed by the Alibaba Cloud Machine Learning Platform (PAI) algorithm team.  The tool aims to assist developers in quickly getting started with Lingjun products and completing the entire development pipeline for LLM, including efficient distributed training, supervised fine-tuning, and offline model inference or verification. It has several merits as follows:

- Support for multiple commonly used LLM such as llama, llama-2, codellama, deepseek, baichuan, qwen, Falcon, GLM, Starcoder, Bloom, chatglm, etc.
- Support for model weight conversion: Mapping operator namespaces between Huggingface, Megatron, and Transformer Engine.
- Support for FP8 training acceleration in Flash Attention 2.0 and Transformer Engine modes, ensuring training convergence.
- Rich and user-friendly usage examples, offering best practices for the entire workflow of LLM pre-training, fine-tuning, evaluation, and inference, as well as reinforcement learning.

## Framework
The design philosophy of Pai-Megatron-Patch is to avoid invasive modifications to the source code of Megatron-LM. In other words, it does not add new modules directly to Megatron-LM. Instead, the functions that need expansion and improvement are presented in the form of patch. This decoupling ensures that users can continue to embrace the best practices of LLM without being affected by upgrades of Megatron-LM.

Pai-Megatron-Patch includes key components for building LLM training, such as model library, tokenizers, model convertors, reinforcement learning , offline text generation, usages examples, and toolkits. The model library provides popular LLMs implemented in Megatron, such as baichuan, bloom, chatglm, falcon, galactica, glm, llama, qwen, and starcoder. More Megatron-based implementations of LLMs will be added as needed in the future. Additionally, the patch provides bidirectional conversion between Huggingface and Megatron model weights. This allows users to easily utilize Huggingface pretrained models for continued pre-training or fine-tuning in Megatron, as well as evaluating model quality using Huggingface's evaluation/inference pipelines on trained Megatron models.

In the reinforcement learning section, the patch offers PPO training workflows, enabling users to perform reinforcement learning with SFT models and RM models. Finally, the patch provides numerous usage examples to help users quickly start LLMs training and offline inference. For specific usage processes within Alibaba Cloud Lingjun products, please refer to the following link: [PAI-Lingjun Intelligent Computing Service LLM solution](https://www.aliyun.com/solution/tech-solution/pai_lingjun).

<div align=center>
<img src=patch.png width=600 height=400 />
</div>


## Technical Reports
- [Sparse Large Model Training Tool Based on Megatron-Core: Best Practices for Alibaba Cloud MoE Large Models](https://mp.weixin.qq.com/s/DkrWEEJ7IxirwWd3qB9Bng)
- [Mixtral-8x7B Training Guide on PAI Lingjun](https://help.aliyun.com/zh/pai/use-cases/train-fine-tune-and-deploy-mixtral-by-using-intelligent-computing-lingjun)
- [Best Practices for the Open Source Model Tongyi Qianwen on PAI Lingjun](https://mp.weixin.qq.com/s?__biz=Mzg4MzgxNDk2OA==&mid=2247491796&idx=1&sn=dc1d719313d794ae1aacdb07669a9545&chksm=cf430783f8348e950218bfcff861a2e6d2d92705807bf5b04f6e9268cc510ffa6e6aa2c87327#rd)
- [Alibaba Cloud Machine Learning PAI Open Source AI Large Model Training Tool Pai-Megatron-Patch, Facilitating the Implementation of Large Model Technologies](https://zhuanlan.zhihu.com/p/655942437)
- [Text Generation and Understanding with the Highest Energy Efficient 27 Billion Parameter GPT Model on a Single Machine](https://zhuanlan.zhihu.com/p/597652820)
- [Chinese Sparse GPT Large Model Implementation — A Key Milestone Towards Low-Cost & High-Performance Multitasking General Natural Language Understanding](https://zhuanlan.zhihu.com/p/561320982)
- [Alibaba Cloud PAI Releases Knowledge Pre-training Tool After Winning the Pre-training Knowledge Measurement Competition](https://zhuanlan.zhihu.com/p/449487792)
- [Alibaba Cloud PAI Wins Dual Championships in FewCLUE Small Sample Learning Based on Large Models](https://developer.aliyun.com/article/788081?spm=a2c6h.12873639.article-detail.17.11c5383cHpFZks&tlog=yuekan_8)


## Contact
Use [Dingtalk](https://www.dingtalk.com/en) to scan blow QR code
<div align=center>
<img src=qr.png width=600 height=450 />
</div>

## License
This project is licensed under the [Apache License (Version 2.0)](https://github.com/alibaba/pai-megatron-patch/blob/master/LICENSE). This toolkit also contains some code modified from other repos under other open-source licenses. See the [NOTICE](https://github.com/alibaba/pai-megatron-patch/blob/master/NOTICE) file for more information.
