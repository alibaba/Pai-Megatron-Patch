# Repo for LLM Supervised Finetune and Text Generation

Date: 2024-01-02

Authors:
- menglibin.mlb


## Usage
### Finetune
**Modifying `use_cache=true` in config.json of the model checkpoint can significantly improve the speed of inference.**  
Run the following command in the container of the image `dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/ngc:23.07-py310-cu121-ubuntu22.04-megatron-patch-llm`
- Llama2
```
$ bash run_ds_train_huggingface_finetune.sh dsw 13B 1 2 1e-5 2048 bf16 2 true llama2-13b true 2 /mnt/llama2-datasets/wudao_train.json /mnt/llama2-datasets/wudao_valid.json /mnt/llama2-ckpts/Llama-2-13b-hf /mnt/output_llama2_finetune
```
- Other model (qwen、chatglm、baichuan2、falcon、bloom......)
```
$ bash run_ds_train_huggingface_finetune.sh dsw 7B 1 2 1e-5 2048 bf16 2 true qwen-7b false 2 /mnt/qwen-datasets/wudao_train.json /mnt/qwen-datasets/wudao_valid.json /mnt/qwen-ckpts/qwen-7b-hf /mnt/output_qwen_7b_finetune
```

### Text Generation
### Text generation by vllm
Run the following command in the container of the image `pai-image-manage-registry.cn-wulanchabu.cr.aliyuncs.com/pai/llm-inference:vllm-0.2.6-v2`
```
$ python text_generation_vllm.py --checkpoint /mnt/llama2-ckpts/Llama-2-13b-chat-hf --input-file /mnt/llama2-datasets/wudao_valid.jsonl --output-file /mnt/llama2-datasets/wudao_valid_output.txt
```
### Text generation by huggingface transformers
```
$ python text_generation_huggingface.py --cuda-visible-devices 0 --checkpoint /mnt/llama2-ckpts/Llama-2-13b-chat-hf --input-file /mnt/llama2-datasets/wudao_valid.jsonl --output-file /mnt/llama2-datasets/wudao_valid_output.txt
```