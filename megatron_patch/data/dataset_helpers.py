# Copyright (c) 2024 Alibaba PAI and Nvidia Megatron-LM Team.
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
import dataclasses
import re
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict

from megatron_patch.data.image_processing import get_visual_transform
import numpy as np
import torch
from torchvision import transforms as T
import json

from megatron.energon import (
    Batch,
    DefaultTaskEncoder,
    VQASample,
)

from megatron_patch.data.energon.chatml import ChatMLSample

from megatron.training import get_args
from megatron_patch.tokenizer import get_tokenizer

# Type for intermediate batch, after batch()
@dataclass
class ImageTaskSample:
    __key__: str
    __subflavors__: Dict
    
    imgs: List[np.ndarray] # (c, h, w)
    videos: List[np.ndarray] # (c, h, w)

    image_thw_grids: np.ndarray
    video_thw_grids: np.ndarray
    image_input_mask: np.ndarray
    video_input_mask: np.ndarray
    second_per_grid_ts: np.ndarray # (n_videos, )

    text: np.ndarray
    target: np.ndarray

# Typing for the resulting batch data after encode_batch()
@dataclass
class VQATaskBatch(Batch):
    __keys__: List[str]
    __subflavors__: List[Dict]
    # (num_tiles, c, h, w)
    imgs: torch.Tensor
    videos: torch.Tensor
    image_thw_grids: torch.Tensor
    video_thw_grids: torch.Tensor
    image_input_mask: torch.Tensor
    video_input_mask: torch.Tensor
    second_per_grid_ts: torch.Tensor # (n_videos, ), read from metadata?

    # (n, seq_len)
    text: torch.Tensor
    # (n, seq_len)
    target: torch.Tensor

class InternalWarning(Warning):
    ...

def convert_to_qwen2vl_content(
        user_input: str, 
        image_pattern: str = '<image>',
        video_pattern: str = '<video>'
    ):
    """
        Split user input into format Qwen2VL tokenizer accepts.
    """
    pattern = r"({image}|{video})".format(image=image_pattern, video=video_pattern)
    contents = []
    cur = 0
    mm_idx = defaultdict(int)
    for matched in re.finditer(pattern, user_input):
        start, end = matched.span()
        if start > cur:
            contents.append({
                "type": "text",
                "text": user_input[cur:start].strip()
            })
            
        contents.append({
            "type": matched.string[start:end][1:-1],
             matched.string[start:end][1:-1]: str(mm_idx[matched.string[start:end][1:-1]])
        })

        cur = end
        mm_idx[matched.string[start:end][1:-1]] += 1

    if cur < len(user_input):
        contents.append({
            "type": "text",
            "text": user_input[cur:len(user_input)].strip()
        })
    
    return contents

class TaskEncoder(DefaultTaskEncoder[Union[VQASample, ChatMLSample], ImageTaskSample, VQATaskBatch, dict]):
    """A simple task encoder for captioning."""

    def __init__(
        self,
    ):
        # Specify the batch_type for default batching (batching is performed here "manually" by
        # overwriting the `batch` method)
        super().__init__()

        self.args = get_args()

        self.tokenizer = get_tokenizer()
        
        self.temporal_patch_size = self.args.temporal_patch_size
        self.merge_size = self.args.spatial_merge_size
        self.patch_size = self.args.patch_size

        self.seq_len = self.args.max_padding_length

    def encode_sample(self, sample: Union[VQASample, ChatMLSample]):
        if isinstance(sample, VQASample):
            is_llava_training = sample.__subflavors__['is_llava_training'] if 'is_llava_training' in sample.__subflavors__ else False
            if is_llava_training:
                raise NotImplementedError('Sample format not supported')
            else:
                yield self.encode_vqa(sample)
        elif isinstance(sample, ChatMLSample):
            yield self.encode_chatml(sample)
        else:
            raise NotImplementedError('Sample format not supported')

    def _flatten_visual_inputs(self, visuals, is_image: bool = True):
        flattened = []
        thw_grids = []
        for visual in visuals:
            if is_image:
                resized_height, resized_width = visual.shape[-2:]
                patches = np.tile(np.array(visual), (self.temporal_patch_size, 1, 1, 1))
            else:
                assert len(visual) % self.temporal_patch_size == 0
                patches = np.array(visual)
                resized_height, resized_width = patches.shape[-2:]

            channel = patches.shape[1]
            grid_t = patches.shape[0] // self.temporal_patch_size
            grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
            patches = patches.reshape(
                grid_t,
                self.temporal_patch_size,
                channel,
                grid_h // self.merge_size,
                self.merge_size,
                self.patch_size,
                grid_w // self.merge_size,
                self.merge_size,
                self.patch_size,
            )
            patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)      
            flatten_patches = patches.reshape(
                grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size
            )     
            flattened.append(flatten_patches)
            thw_grids.append((grid_t, grid_h, grid_w))
        return flattened, np.array(thw_grids)

    def encode_chatml(self, sample: ChatMLSample):
        # TODO: modify get_visual_transform to add more augmentations
        imgs = [get_visual_transform(img)[0] for img in sample.imgs]
        videos = [[get_visual_transform(frame)[0] for frame in video] for video in sample.videos]
        # NOTE: make n_frames even foreach video
        for i, video in enumerate(videos):
            videos[i] = video[:len(video) // 2 * 2]

        # NOTE: flatten all images
        flattened_imgs, image_thw_grids = self._flatten_visual_inputs(imgs, is_image=True)
        flattened_videos, video_thw_grids = self._flatten_visual_inputs(videos, is_image=False)

        # NOTE: generate qwen2vl conversations
        conversation = json.loads(sample.conversation) if isinstance(sample.conversation, (str, bytes)) else sample.conversation
        second_per_grid_ts = [1 / 2.0] * len(video_thw_grids)
        if 'conversations' in conversation:
            second_per_grid_ts = conversation.get('second_per_grid_ts', second_per_grid_ts)
            second_per_grid_ts = [float(i) for i in second_per_grid_ts]
            conversation = conversation['conversations']
 
        role_key = 'from' if 'from' in conversation[0] else 'role'
        content_key = 'value' if 'from' in conversation[0] else 'content'
        
        # NOTE: assume the conversation format is: [System]? (User Assistant)+
        converted_conversation = []
        if len(conversation) % 2 == 0:
            # Default Prompt
            converted_conversation.append({
                'role': 'system',
                'content': 'You are a helpful assistant.'
            })
        else:
            converted_conversation.append({
                'role': 'system',
                'content': conversation[0][content_key]
            })
            conversation = conversation[1:]
        
        EXPECTED_ROLE = ['human', 'gpt']
        for turn_idx, turn in enumerate(conversation):
            role = turn[role_key]
            if role != EXPECTED_ROLE[turn_idx % len(EXPECTED_ROLE)]:
                raise InternalWarning(f"Expect conversation organized in order: [sys] human gpt human gpt..., but got role '{role}' in turn {turn_idx}")
            content = turn[content_key]

            if role == 'human':
                role = 'user'
                content = convert_to_qwen2vl_content(content)
            elif role == 'gpt':
                role = 'assistant'
            
            converted_conversation.append({
                'role': role,
                'content': content
            })
        conversation = converted_conversation

        # NOTE: we need to mask all system/user input tokens and assistant generation prefix tokens
        input_ids = self.tokenizer.apply_chat_template(conversation, tokenize=True, return_tensors="np")[0]
        target = input_ids.copy()

        system_prompt_prefix = len(self.tokenizer.apply_chat_template([conversation[0]], tokenize=True))
        assistant_generation_prefix = 3
        pad_token_id = self.tokenizer.pad_token_id

        target[:system_prompt_prefix] = pad_token_id
        offset = system_prompt_prefix
        for turn_idx, turn in enumerate(conversation[1:]):
            turn_tokens = self.tokenizer.apply_chat_template([turn], tokenize=True, return_tensors="np")[0]
            turn_content = turn_tokens[system_prompt_prefix:]
            n_tokens = len(turn_content)
            if (target[offset: offset + n_tokens] != turn_content).any():
                raise InternalWarning("Encode Error")

            if turn['role'] == 'user':
                target[offset: offset + n_tokens] = pad_token_id
            elif turn['role'] == 'assistant':
                target[offset: offset + assistant_generation_prefix] = pad_token_id   
            offset += n_tokens

        # NOTE: expand image_pad & video_pad
        merge_length = self.merge_size**2
        image_token_id, video_token_id = self.tokenizer.encode(['<|image_pad|>', '<|video_pad|>'])

        image_token_indices = np.where(input_ids == image_token_id)[0]
        assert len(image_token_indices) == len(image_thw_grids), f"With {len(image_thw_grids)} images in the sample, but {len(image_token_indices)} image placeholders!"
        video_token_indices = np.where(input_ids == video_token_id)[0]
        assert len(video_token_indices) == len(video_thw_grids), f"With {len(video_thw_grids)} images in the sample, but {len(video_token_indices)} video placeholders!"
        image_thw_grids, video_thw_grids = np.array(image_thw_grids, dtype=np.int64), np.array(video_thw_grids, dtype=np.int64)

        target_length = (
            input_ids.shape[0] 
            - image_thw_grids.shape[0] + image_thw_grids.prod(axis=-1).sum() // merge_length
            - video_thw_grids.shape[0] + video_thw_grids.prod(axis=-1).sum() // merge_length
        )
        if target_length > self.seq_len:
            raise InternalWarning(f"Long sequence with length {target_length} found, dropped...")
        final_input_ids = np.zeros(target_length, dtype=input_ids.dtype)
        final_input_masks = final_input_ids.copy()

        image_idx, video_idx = 0, 0
        indices = np.sort(np.concatenate([image_token_indices, video_token_indices]))

        cur_x, cur_y = 0, 0
        for idx in indices:
            token_id = input_ids[idx]
            if token_id == image_token_id:
                size = image_thw_grids[image_idx].prod() // merge_length
                image_idx += 1
            elif token_id == video_token_id:
                size = video_thw_grids[video_idx].prod() // merge_length
                video_idx += 1
            # NOTE:
            # input_ids[cur_x:idx] -> final_input_ids[cur_y:cur_y + idx - cur_x]
            # input_ids[idx] -> final_input_ids[cur_y + idx - cur_x: cur_y + idx - cur_x + size]
            final_input_ids[cur_y: cur_y + idx - cur_x] = input_ids[cur_x:idx]
            final_input_masks[cur_y: cur_y + idx - cur_x] = target[cur_x:idx]
            cur_y += idx - cur_x
            final_input_ids[cur_y: cur_y + size] = token_id
            final_input_masks[cur_y: cur_y + size] = pad_token_id
            cur_y += size
            cur_x = idx + 1
        
        if cur_x < len(input_ids):
            final_input_ids[cur_y:] = input_ids[cur_x:]
            final_input_masks[cur_y:] = target[cur_x:]

        target = np.roll(final_input_masks, shift=-1)
        target[-1] = pad_token_id

        if (target == pad_token_id).all():
            raise InternalWarning("Sample with all masked label, dropped.")

        image_input_mask = final_input_ids == self.tokenizer.image_token_id
        video_input_mask = final_input_ids == self.tokenizer.video_token_id
        # collect data
        return ImageTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            imgs=flattened_imgs,
            videos=flattened_videos,

            image_thw_grids=image_thw_grids,
            video_thw_grids=video_thw_grids,
            second_per_grid_ts = np.array(second_per_grid_ts, dtype=np.float32),

            image_input_mask=image_input_mask,
            video_input_mask=video_input_mask,

            text=final_input_ids,
            target=target,
        )
    
    def encode_vqa(self, sample: VQASample):
        augment = sample.__subflavors__['augmentation'] if 'augmentation' in sample.__subflavors__ else False
        has_video = sample.__subflavors__['has_video'] if 'has_video' in sample.__subflavors__ else False

        if has_video:
            raise NotImplementedError("You should use sharegpt dataset to train with videos.")
        else:
            # TODO: add args
            imgs = get_visual_transform(sample.image)
            flatten_patches, thw_grids = self._flatten_visual_inputs(imgs, is_image=True)

        assert "<image>" in sample.context # ?
        # NOTE: we expect a context is a string with <image> conetnt
        if isinstance(sample.answers, list):
            answer_list = sample.answers
            weight_list = np.array(sample.answer_weights).astype(np.float32)
            weight_list = weight_list / np.sum(weight_list)
            answer_idx = np.random.choice(weight_list.shape[0], 1, p=weight_list)[0]
            answer = answer_list[answer_idx]
        else:
            answer = sample.answers

        conversation = [
            {"role": "user", "content": convert_to_qwen2vl_content(sample.context)},
            {"role": "assistant", "content": answer},
        ]

        user_inputs = self.tokenizer.apply_chat_template(conversation[:-1], tokenize=False)
        text = self.tokenizer.apply_chat_template(conversation, tokenize=False)

        # text, target = self.tokenizer.tokenize_conversation(conversation, False, False)
        # replace <image> token by <image> * (thw)
        merge_length = self.merge_size**2
        image_token = '<|image_pad|>'
        assert len(thw_grids) == 1, "Only one image per sample is supported!"
        index = 0
        while image_token in text:
            grid_t, grid_h, grid_w = thw_grids[index]
            l = grid_t * grid_h * grid_w
            text = text.replace(
                image_token, "<|placeholder|>" * (l // merge_length), 1
            )
            user_inputs = user_inputs.replace(
                image_token, "<|placeholder|>" * (l // merge_length), 1
            )
            index += 1
        text = text.replace("<|placeholder|>", image_token)
        user_inputs = user_inputs.replace("<|placeholder|>", image_token)

        input_ids = self.tokenizer.tokenize(text)
        user_input_ids = self.tokenizer.tokenize(user_inputs)
        if len(input_ids) > self.seq_len:
            raise InternalWarning(f"Long sequence with length {len(input_ids)} found, dropped...")
        
        target = np.array(input_ids[1:] + [self.tokenizer.pad_token_id])
        if len(user_input_ids) >= len(input_ids):
            raise InternalWarning(f"Sample not supported, dropped...")
        # ensure user inputs is a prefix of full text
        if not (np.array(user_input_ids) == np.array(input_ids[:len(user_input_ids)])).all():
            raise InternalWarning(f"Sample not supported, dropped...")
        # mask input
        target[:len(user_input_ids)-1] = self.tokenizer.pad_token_id

        img_token_id = self.tokenizer.image_token_id
        image_input_mask = np.array(input_ids) == img_token_id

        # collect data
        return ImageTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,

            imgs=flatten_patches,
            videos=list(),

            image_thw_grids=thw_grids,
            video_thw_grids=torch.empty([0, 3], dtype=torch.long),

            image_input_mask=image_input_mask,
            video_input_mask=None,
            second_per_grid_ts=np.zeros(0, dtype=np.float32),
            
            text=input_ids,
            target=target,
        )

    def batch(self, samples: List[ImageTaskSample]) -> VQATaskBatch:
        # Stack images to [num_tiles, c, h, w]. If there are no images (text-only), then use a dummy image.
        imgs = [img for s in samples for img in s.imgs]
        if len(imgs) > 0:
            imgs = torch.cat([torch.from_numpy(img) for img in imgs])
        else:
            imgs = torch.empty([0, 3 * self.temporal_patch_size * self.patch_size * self.patch_size], dtype=torch.float32)
        
        image_thw_grids = [thw_grids for s in samples for thw_grids in s.image_thw_grids]
        if len(image_thw_grids) > 0:
            image_thw_grids = torch.from_numpy(np.array(image_thw_grids)).long()
            assert image_thw_grids.prod(dim=-1).sum() == imgs.shape[0]
        else:
            image_thw_grids = torch.empty([0, 3], dtype=torch.long)
        
        # Stack videos to [num_tiles, c, h, w]. If there are no videos (text-only), then use a dummy video.
        videos = [video for s in samples for video in s.videos]
        if len(videos) > 0:
            videos = torch.cat([torch.from_numpy(video) for video in videos])
        else:
            videos = torch.empty([0, 3 * self.temporal_patch_size * self.patch_size * self.patch_size], dtype=torch.float32)
        
        second_per_grid_ts = [second_per_grid for s in samples for second_per_grid in s.second_per_grid_ts]
        if len(second_per_grid_ts) > 0:
            second_per_grid_ts = torch.from_numpy(np.array(second_per_grid_ts)).float()
        else:
            second_per_grid_ts = torch.empty([0, ], dtype=torch.float32)
        
        video_thw_grids = [thw_grids for s in samples for thw_grids in s.video_thw_grids]
        if len(video_thw_grids) > 0:
            video_thw_grids = torch.from_numpy(np.array(video_thw_grids)).long()
            assert video_thw_grids.prod(dim=-1).sum() == videos.shape[0]
        else:
            video_thw_grids = torch.empty([0, 3], dtype=torch.long)

        # If the user hasn't defined a target sequence length, then use the max along the sample lengths.
        max_seq_len = self.seq_len
        if not max_seq_len:
            max_seq_len = max(len(s.text) for s in samples)

        text_mat = np.full((len(samples), max_seq_len), self.tokenizer.pad_token_id, dtype=np.int64)
        # +1 to accommodate shift to left by one later.
        target_mat = np.full((len(samples), max_seq_len), self.tokenizer.pad_token_id, dtype=np.int64)
        
        image_input_masks = np.zeros_like(text_mat, dtype=bool)
        video_input_masks = np.zeros_like(text_mat, dtype=bool)
        for i, s in enumerate(samples):
            # If the sample/target length exceeds the target sequence length, then truncate.
            text_len = min(max_seq_len, len(s.text))
            target_len = min(max_seq_len, len(s.target))

            text_mat[i, :text_len] = np.array(s.text)[:text_len]
            # NOTE: we should assert user input sequence will not be truncated
            if s.image_input_mask is not None:
                image_input_masks[i, :text_len] = np.array(s.image_input_mask)[:text_len]
            if s.video_input_mask is not None:
                video_input_masks[i, :text_len] = np.array(s.video_input_mask)[:text_len]
            target_mat[i, :target_len] = np.array(s.target)[:target_len]
        
        batch = VQATaskBatch(
            __keys__=[s.__key__ for s in samples],
            __subflavors__=[s.__subflavors__ for s in samples],
            imgs=imgs,
            videos=videos,
            image_thw_grids=image_thw_grids,
            video_thw_grids=video_thw_grids,
            second_per_grid_ts=second_per_grid_ts,
            image_input_mask=torch.from_numpy(image_input_masks),    
            video_input_mask=torch.from_numpy(video_input_masks),
            text=torch.from_numpy(text_mat),
            target=torch.from_numpy(target_mat),
        )

        return batch

    def encode_batch(self, batch: VQATaskBatch) -> dict:
        raw = dataclasses.asdict(batch)
        del raw["__subflavors__"]
        return raw


def print_error_handler(exc: Exception, key: Optional[str], debug=False):
    if not debug and isinstance(exc, InternalWarning):
        return
    print(
        f"The following exception occurred in the dataloader for sample {key} and is skipped",
        file=sys.stderr,
    )
    traceback.print_exc()
