## hf-to-megatron
hf-to-megatron是一款模型ckpt转换工具，方便用户低门槛的将huggingface版的ckpt转换到megatron格式，以使用megatron-lm的分布式能力训练LLM大模型。转换后的模型需配合PAI-Megatron-Patch代码库使用。目前已经支持下列模型：

+ bloom
+ llama/alpaca
+ chatglm
+ galactica
+ glm
+ glm130B
+ falcon
+ starcoder

相关转换后的模型存放在：oss://atp-modelzoo/release/models/pai-megatron-patch/
