## hf-to-megatron
hf-to-megatron is a model checkpoint conversion tool designed to easily convert Hugging Face checkpoints to the Megatron format. This conversion allows users to leverage the distributed capabilities of Megatron-LM for training large language models (LLMs). The converted models must be used in conjunction with the PAI-Megatron-Patch codebase. The tool currently supports the following models:

+ bloom
+ llama/alpaca
+ chatglm
+ galactica
+ glm
+ glm130B
+ falcon
+ starcoder

The converted models are stored at: oss://atp-modelzoo/release/models/pai-megatron-patch/