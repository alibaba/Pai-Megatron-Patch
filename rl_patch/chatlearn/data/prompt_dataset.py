# Copyright 2024 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import copy
from collections import defaultdict
from typing import List, Dict
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class PromptPipeline(Dataset):
    """
    a dataset of list of no padded prompt tensors
    truncted to max_prompt_length from right
    """

    def __init__(self, prompts: List[str], max_prompt_length: int, tokenizer=None):
        super().__init__()

        for p in prompts:
            assert len(p) > 0, "Got empty prompt"
        assert max_prompt_length > 0, \
            "Prompt length for RLHF/OnlineDPO/GRPO trainer must be an integer greater than 0"

        if len(prompts[0]) == 3:
            prompt_encodings = [tokenizer.tokenize(prompt)[:max_prompt_length] for prompt, _, _ in prompts]
        else:
            prompt_encodings = [tokenizer.tokenize(prompt)[:max_prompt_length] for prompt in prompts]
        prompt_id_tensors = [torch.tensor(p_encoding, dtype=torch.long) for p_encoding in prompt_encodings]

        # dup dataset if num_inference_per_prompt
        self.data = []
        prompts = [{"input_ids": prompt_tensor} for prompt_tensor in prompt_id_tensors]
        for p in prompts:
            dup = [copy.deepcopy(p) for i in range(1)]
            self.data.extend(dup)

        self.tokenizer = tokenizer

    def __getitem__(self, ix: int):
        return self.data[ix]

    def __len__(self) -> int:
        return len(self.data)

    def collate_fn(self, samples):
        collate_dict = defaultdict(list)

        # Loop over the samples and append each tensor value to the corresponding list
        for sample in samples:
            for key in sample.keys():
                collate_dict[key].append(sample[key])

        # Return the collate_dict
        return collate_dict


class VLLMPromptPipeline(Dataset):
    """
    process this format
    {
        "data_source": data_source,
        "prompt": [{
            "role": "user",
            "content": question,
        }],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": solution
        },
        "extra_info": {
            'split': split,
            'index': idx,
            'answer': answer_raw,
            "question": question_raw,
        }
    }
    self.data format
    {"input_ids": prompt_ids, "prompt": prompt}
    """

    def __init__(self, data_list: List[Dict], seq_length: int, tokenizer: AutoTokenizer = None,
                 num_inference_per_prompt: int = 1):  # pylint: disable=super-init-not-called
        super().__init__()

        self.tokenizer = tokenizer
        self.data = []

        for data_item in data_list:
            prompt = data_item["prompt"]
            data_source = data_item.get("data_source", "")
            ground_truth = data_item['reward_model']['ground_truth']
            if isinstance(prompt, list):
                prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            input_ids = self.tokenizer.encode(prompt)
            processed_data = {"input_ids": input_ids, "prompt": prompt, "data_source": data_source,
                              "ground_truth": ground_truth}
            if seq_length > len(input_ids):
                self.data.extend([processed_data] * num_inference_per_prompt)

    def __getitem__(self, ix: int):
        return self.data[ix]

    def __len__(self) -> int:
        return len(self.data)

    def collate_fn(self, samples):
        collate_dict = defaultdict(list)

        # Loop over the samples and append each tensor value to the corresponding list
        for sample in samples:
            for key in sample.keys():
                collate_dict[key].append(sample[key])

        # Return the collate_dict
        return collate_dict