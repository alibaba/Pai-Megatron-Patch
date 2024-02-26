import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# copy config.json and llama_moe.py to hugging face model_path 

model_path = '/workdir/llama/llama2_7b_mcore_moe4_tp2_pp1_ep4_hf'

config = AutoConfig.from_pretrained(
    model_path,
    trust_remote_code=True,
    num_local_experts=4, # cover param in config.json
    num_experts_per_tok=1,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    padding_side="left",
    trust_remote_code=True,
)
print(f'config {config}')
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=config,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
print(f'model {model}')

input_ids = torch.tensor([[1,2,3]]).long().cuda()
model.cuda()
with torch.inference_mode():
    output = model(input_ids=input_ids)

print(f'model output {output}')