import argparse
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration

def inference(HF_PATH):

    hf_transformer_config = AutoConfig.from_pretrained(HF_PATH)

    if hf_transformer_config.architectures[0] == "Qwen3VLForConditionalGeneration":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            HF_PATH, dtype="auto", device_map="auto"
        )
    elif hf_transformer_config.architectures[0] == "Qwen3VLMoeForConditionalGeneration":
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            HF_PATH, dtype="auto", device_map="auto"
        )

    processor = AutoProcessor.from_pretrained(HF_PATH)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    inference(args.model_path)