## Introduction
English | [简体中文](./README_zh-CN.md)

Pai-Megatron-Patch tool is developed by the PAI team of the Alibaba Cloud.
It aims to help LLM developers quickly get started with efficient large distributed training via PAI-Lingjun Intelligent Computing Service.
This project leverage [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) to build LLM pretraining, finetuning, evalultion and inference pipeline. 
It contains megatron models such as llama, baichuan, bloom, chatglm, falcon, galactica, glm, qwen, starcoder etc. 
We also provide model ckpt conversion between huggingface and megatron.
see the [PAI-Lingjun Intelligent Computing Service LLM solution](https://help.aliyun.com/document_detail/2505831.html?spm=5176.28352543.J_9l_YP1wy4J7aEdtojTyUD.1.347850adeLHhmP&tab=onestop) for more information. 

<div align=center>
<img src=patch_en.png width=600 height=400 />
</div>

## Quick Start

[Environment Preparation](https://help.aliyun.com/document_detail/2505831.html?spm=5176.28352543.J_9l_YP1wy4J7aEdtojTyUD.1.347850adeLHhmP&tab=onestop)

[Data Preparation](toolkits/pretrain_data_preprocessing/README.md)

[Huggingface Training](examples/hfds.md)

[Megatron Training](examples/megatron.md)

[RLHF](rlhf/README.md)

[Inference](megatron_patch/generation/megatron.md)


## License
This project is licensed under the [Apache License (Version 2.0)](https://github.com/alibaba/pai-megatron-patch/blob/master/LICENSE). This toolkit also contains some code modified from other repos under other open-source licenses. See the [NOTICE](https://github.com/alibaba/pai-megatron-patch/blob/master/NOTICE) file for more information.
