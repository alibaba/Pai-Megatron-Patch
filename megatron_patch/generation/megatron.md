## Megatron推理
此处复用了Megatron-LM中的推理框架。  
改动：  
- 修改tokenizer处理数据的部分，适配huggingface的tokenizer  
- 增加推理过程中对重复生成的处理，支持repetition_penalty.  