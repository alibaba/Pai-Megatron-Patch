from qwen_omni_utils import process_mm_info
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeThinkerForConditionalGeneration

MODEL_PATH = "/mnt/data_2/ckpts/huggingface/Qwen3-Omni-30B-A3B-Instruct"
config = AutoConfig.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )
thinker = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(MODEL_PATH)
processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/mnt/data_2/jerry.lp/Pai-Megatron-Patch-jerryli1981/toolkits/multimodal_data_preprocessing/australia.jpg"},
            {"type": "audio", "audio": "/mnt/data_2/jerry.lp/Pai-Megatron-Patch-jerryli1981/toolkits/multimodal_data_preprocessing/glass-breaking-151256.mp3"},
            {"type": "audio", "audio": "/mnt/data_2/jerry.lp/Pai-Megatron-Patch-jerryli1981/y2293.wav"},
            {"type": "text", "text": "What can you see and hear? Answer in one short sentence."}
        ],
    },
]

# Set whether to use audio in video
USE_AUDIO_IN_VIDEO = False

# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, 
                   audio=audios, 
                   images=images, 
                   videos=videos, 
                   return_tensors="pt", 
                   padding=True, 
                   use_audio_in_video=USE_AUDIO_IN_VIDEO)

# Generate
generation = thinker.generate(**inputs, max_new_tokens=2048)
generate_ids = generation[:, inputs.input_ids.size(1):]
response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(response)