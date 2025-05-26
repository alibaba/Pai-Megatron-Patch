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

def pad_to_max_len(all_tokens_right_padded, max_len, pad_value):
    pad_length = max_len - all_tokens_right_padded.size(1)
    if pad_length <= 0:
        return all_tokens_right_padded
    # Pad the tensor with zeros on the right side to the desired length
    padded_tensor = torch.nn.functional.pad(all_tokens_right_padded, (0, pad_length), mode='constant', value=pad_value)
    return padded_tensor

def generate_loss_mask_position_ids(tokens: torch.Tensor, prompt_token_length: list, response_token_length:list):
    # Setup attention mask by prompt token length and response token length
    loss_mask = torch.zeros_like(tokens, dtype=torch.int32, device=tokens.device)
    for i in range(len(prompt_token_length)):
        loss_mask[i, prompt_token_length[i]: prompt_token_length[i] + response_token_length[i]] = 1.0
    _, seq_len = tokens.size()
    position_ids = torch.arange(seq_len, dtype=torch.long, device=tokens.device)
    position_ids = position_ids.unsqueeze(0).expand_as(tokens)

    return loss_mask, position_ids