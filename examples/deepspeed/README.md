# Repo for LLM Supervised Finetune and Text Generation

Date: 2024-01-02

Authors:
- menglibin.mlb


## Finetune
**Modifying `use_cache=true` in config.json of the model checkpoint can significantly improve the speed of inference.**  
Run the following command in the container of the image `dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/ngc:23.07-py310-cu121-ubuntu22.04-megatron-patch-llm`
### Usage
```
$ bash run_ds_train_huggingface_finetune.sh --help
```
Usage: bash run_ds_train_huggingface_finetune.sh \
    [--env ENV] \
    [--model-size MODEL_SIZE] \
    [--micro-batch-size MICRO_BATCH_SIZE] \
    [--gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS] \
    [--learning-rate LEARNING_RATE] \
    [--sequence-length SEQUENCE_LENGTH] \
    [--precision PRECISION] \
    [--zero-stage ZERO_STAGE] \
    [--enable-gradient-checkpointing ENABLE_GRADIENT_CHECKPOINTING] \
    [--model-name MODEL_NAME {llama2-13b, qwen-7b, qwen-14b, qwen-72b}] \
    [--flash-attention FLASH_ATTENTION] \
    [--epoch EPOCH] \
    [--train-dataset TRAIN_DATASET] \
    [--validation-dataset VALIDATION_DATASET] \
    [--pretrain-model-path PRETRAIN_MODEL_PATH] \
    [--finetune-output-path FINETUNE_OUTPUT_PATH]
### Demo
#### Llama2
```
$ bash run_ds_train_huggingface_finetune.sh --env dsw --model-size 13B --micro-batch-size 1 --gradient-accumulation-steps 2 --learning-rate 1e-5 --sequence-length 2048 --precision bf16 --zero-stage 2 --enable-gradient-checkpointing true --model-name llama2-13b --flash-attention true --epoch 2 --train-dataset /mnt/llama2-datasets/wudao_train.json --validation-dataset /mnt/llama2-datasets/wudao_valid.json --pretrain-model-path /mnt/llama2-ckpts/Llama-2-13b-hf --finetune-output-path /mnt/output_llama2_finetune
```
#### Other model (qwen、chatglm、baichuan2、falcon、bloom......)
```
$ bash run_ds_train_huggingface_finetune.sh --env dsw --model-size 7B --micro-batch-size 1 --gradient-accumulation-steps 2 --learning-rate 1e-5 --sequence-length 2048 --precision bf16 --zero-stage 2 --enable-gradient-checkpointing true --model-name qwen-7b --flash-attention false --epoch 2 --train-dataset /mnt/qwen-datasets/wudao_train.json --validation-dataset /mnt/qwen-datasets/wudao_valid.json --pretrain-model-path /mnt/qwen-ckpts/qwen-7b-hf --finetune-output-path /mnt/output_qwen_7b_finetune
```

## Text Generation
Run the following command in the container of the image `pai-image-manage-registry.cn-wulanchabu.cr.aliyuncs.com/pai/llm-inference:vllm-0.2.6-v2`
### Usage
```
$ python text_generation_huggingface.py --help
```
Usage: text_generation_huggingface.py \
    --checkpoint CHECKPOINT \
    --input-file INPUT_FILE \ 
    --output-file OUTPUT_FILE \
    [--cuda-visible-devices CUDA_VISIBLE_DEVICES] \
    [--output-max-tokens OUTPUT_MAX_TOKENS]
```
$ python text_generation_vllm.py --help
```
Usage: text_generation_vllm.py \
    --checkpoint CHECKPOINT \
    --input-file INPUT_FILE \
    --output-file OUTPUT_FILE \
    [--tensor-parallel-size TENSOR_PARALLEL_SIZE] \
    [--output-max-tokens OUTPUT_MAX_TOKENS]
### Demo
#### Text generation by vllm
```
$ python text_generation_vllm.py --checkpoint /mnt/llama2-ckpts/Llama-2-13b-chat-hf --input-file /mnt/llama2-datasets/wudao_valid.jsonl --output-file /mnt/llama2-datasets/wudao_valid_output.txt --tensor-parallel-size 1
```
#### Text generation by huggingface transformers
```
$ python text_generation_huggingface.py --checkpoint /mnt/llama2-ckpts/Llama-2-13b-chat-hf --input-file /mnt/llama2-datasets/wudao_valid.jsonl --output-file /mnt/llama2-datasets/wudao_valid_output.txt --cuda-visible-devices 0
```