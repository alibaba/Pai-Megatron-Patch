## Quick Start


|             |                                                      Megatron-Core                                                       |                                                                                        ChatLearn                                                                                        |    verl     |
|:------------|:------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------:|
|Qwen3       |[ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/qwen3/README.md)|[ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/qwen3/README_grpo.md) | Coming Soon |
|QwQ         |[ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/qwq/README.md)| N/A | N/A |
|Qwen2.5-VL  |[ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/qwen2_5_vl/README.md)| N/A | N/A |
|Moonlight   |[ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/moonlight/README.md)|[ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/moonlight/README_grpo.md)| N/A |
|DeepSeek-V3 |[ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/deepseek_v3/README.md)| N/A | N/A |
|DeepSeek-R1 | N/A |[ReadMe](https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/deepseek_v3/README_grpo.md)| Coming Soon |



## Introduction
English | [简体中文](./README_zh-CN.md)

Pai-Megatron-Patch (https://github.com/alibaba/Pai-Megatron-Patch) is a deep learning training toolkit built for developers to train and predict LLMs & VLMs by using Megatron framework easily. With the continuous development of LLMs, the model structure and scale are rapidly evolving. Although these models can be conveniently manufactured using Transformers or DeepSpeed training framework, the training efficiency is comparably low. This phenomenon becomes even severer when the model scale exceeds 10 billion. The primary objective of Pai-Megatron-Patch is to effectively utilize the computational power of GPUs for LLM. This tool allows convenient training of commonly used LLM with all the accelerating techniques provided by Megatron-LM.

What's New:
- **Support all Mcore Models Context Parallel Reinforcement Training via ChatLearn** [🔥🔥 2025.08.31]
- **Support all Mcore Models GSPO Reinforcement Training via ChatLearn** [🔥🔥 2025.08.11]
- **Support DeepSeek-V3-671B GRPO Reinforcement Training using Megatron-Core and ChatLearn** [🔥🔥 2025.07.31]
- **Support Qwen3-235B GRPO Reinforcement Training using Megatron-Core and ChatLearn** [🔥🔥 2025.07.20]
- **Support Moonlight GRPO Reinforcement Training using Megatron-Core and ChatLearn** [🔥🔥 2025.06.30]
- **Support Qwen3 GRPO Reinforcement Training using Megatron-Core and ChatLearn** [🔥🔥 2025.06.03]
- **Support Qwen2.5 GRPO Reinforcement Training using Megatron-Core and ChatLearn** [🔥🔥 2025.05.18]
- **Support all Qwen3 Training with torch_dist checkpoint** [🔥🔥 2025.04.29]
- **[Experimental]Support distributed checkpoint conversion for large LLM** [🔥🔥 2025.04.16]
- **Upgrade DeepSeek-V3 SFT with fully Mcore implementation.** [🔥🔥 2025.03.31]
- **Support training QwQ by using Megatron-Core.** [🔥🔥 2025.03.27]
- **Support training Qwen2.5-VL by using Megatron-Core.** [🔥🔥 2025.03.21]
- **Support training Moonlight-16B-A3B from Moonshot AI KIMI by using Megatron-Core.** [🔥🔥 2025.03.14]
- **Optimize Checkpoint Conversion of DeepSeek-V3 and add support training with ETP** [🔥🔥 2025.03.14]
- **Support training DeepSeek-V3 671B model by using Megatron-Core.** [🔥🔥 2025.02.21]
- **Upgrade LLM SFT Training Process** [🔥🔥 2025.02.20]
- **Upgrade DeepSeek-V2-MoE for facilitating a smooth transition to integrating the DeepSeek-V3-MoE.** [🔥🔥 2025.01.16]
- **Upgrade Qwen2-VL models to support Sequence Parallel, VPP and TP-Comm-Overlap.** [🔥🔥 2025.01.15]
- **Upgrade Qwen2-VL models to support MG2HF ckpts conversion and training with multi-turn complex multimodal samples.** [🔥🔥 2024.12.27]
- **Support training Qwen2-VL models by using Megatron-Core.** [🔥🔥 2024.11.27]
- **Support training LLaVA models by using Megatron-Core.** [🔥🔥 2024.11.20]
- **Add llm auto configurator and apply per seq sft loss for qwen2/2.5 models.** [🔥🔥 2024.10.30]
- **Upgrade deepseek-v2-moe models to support MLA via transformer engine and pipeline ckpts conversion.** [🔥🔥 2024.09.26]
- **Support training Qwen2.5 models by using Megatron-Core.** [🔥🔥 2024.09.20]
- **Support Sequence Packing in SFT for Qwen2 and LLaMA 3.1 models.** [🔥🔥 2024.09.13]
- **Upgrade qwen2 dense and moe models to support Flash-Attention 3, Offloading, Comm-Overlapping features.** [🔥🔥 2024.08.26]
- **Support training LLaMA 3.1 dense models with Flash-Attention 3 backend.** [🔥🔥 2024.08.23]
- **Support training LLaMA 3.1 dense models by using Megatron-Core.** [🔥🔥 2024.08.23]
- **Support auto optimizer offloading in OffloadDistributedOptimizer.** [🔥🔥 2024.07.25]
- **Support static optimizer offloading in OffloadDistributedOptimizer.** [🔥🔥 2024.07.15]
- **Support training qwen2 moe models by using Megatron-Core.** [🔥🔥 2024.06.19]
- **Support training qwen2 dense models by using Megatron-Core.** [🔥🔥 2024.06.12]
- **Support training deepseek-v2-moe models by using Megatron-Core.** [🔥🔥 2024.05.30]
- **Support training qwen1.5-moe models by using Megatron-Core.** [🔥🔥 2024.05.13]
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
- [基于PAI-ChatLearn的GSPO强化学习实践](https://mp.weixin.qq.com/s/ODl1_yZk-cJdBdE7TwLAZA)
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


## Contact
Use [Dingtalk](https://www.dingtalk.com/en) to scan blow QR code.

Note: group 1 and 2 is full, please add group 3.  
<div align=center>
<img src=qr.png width=600 height=450 />
</div>

<div align=center>
<img src=qr2.png width=600 height=450 />
</div>

<div align=center>
<img src=qr3.png width=600 height=450 />
</div>

## License
This project is licensed under the [Apache License (Version 2.0)](https://github.com/alibaba/pai-megatron-patch/blob/master/LICENSE). This toolkit also contains some code modified from other repos under other open-source licenses. See the [NOTICE](https://github.com/alibaba/pai-megatron-patch/blob/master/NOTICE) file for more information.
