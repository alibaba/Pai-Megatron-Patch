# TODO: Add a License
import dataclasses
import re
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict

from image_processing import get_visual_transform
import numpy as np
import torch
from torchvision import transforms as T

from megatron.energon import (
    Batch,
    DefaultTaskEncoder,
    VQASample,
)

from megatron.training import get_args
from megatron_patch.tokenizer import get_tokenizer

# Type for intermediate batch, after batch()
@dataclass
class ImageTaskSample:
    __key__: str
    __subflavors__: Dict
    # (c, h, w)
    imgs: List[np.ndarray]
    image_thw_grids: List[Tuple[int]]
    video_thw_grids: List[Tuple[int]]
    image_input_mask: np.ndarray
    video_input_mask: np.ndarray
    text: np.ndarray
    target: torch.Tensor = None

# Typing for the resulting batch data after encode_batch()
@dataclass
class VQATaskBatch(Batch):
    __keys__: List[str]
    __subflavors__: List[Dict]
    # (num_tiles, c, h, w)
    imgs: torch.Tensor
    image_thw_grids: torch.Tensor
    video_thw_grids: torch.Tensor
    image_input_mask: torch.Tensor
    video_input_mask: torch.Tensor
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

class TaskEncoder(DefaultTaskEncoder[VQASample, ImageTaskSample, VQATaskBatch, dict]):
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

    def encode_sample(self, sample: VQASample):
        if isinstance(sample, VQASample):
            is_llava_training = sample.__subflavors__['is_llava_training'] if 'is_llava_training' in sample.__subflavors__ else False
            if is_llava_training:
                raise NotImplementedError('Sample format not supported')
            else:
                yield self.encode_vqa(sample)
        else:
            raise NotImplementedError('Sample format not supported')

    def encode_vqa(self, sample: VQASample):
        augment = sample.__subflavors__['augmentation'] if 'augmentation' in sample.__subflavors__ else False
        has_video = sample.__subflavors__['has_video'] if 'has_video' in sample.__subflavors__ else False

        if has_video:
            # Grab the selected frames of the video as a tensor with shape
            # fhwc: (num_frames, height, width, num_channels).
            # video_fhwc = sample.image.permute(0, 2, 3, 1)
            # selected_frames = torch.linspace(
            #     0, video_fhwc.shape[0] - 1, self.args.num_frames).long()
            # video_frame_fhwc = video_fhwc[selected_frames]
            # imgs = []
            # for video_frame_hwc in video_frame_fhwc:
            #     imgs += get_visual_transform(
            #         video_frame_hwc, self.img_h, self.img_w,
            #         self.args.use_tiling, self.args.max_num_tiles,
            #         self.args.use_thumbnail, augment=False)
            raise NotImplementedError()
        else:
            # TODO: add args
            imgs = get_visual_transform(
                sample.image
            )
            resized_height, resized_width = imgs[0].shape[-2:]
            # shape: c x img_h x img_w
            # split single image into tiles for dynamic resolution
            patches = np.tile(np.array(imgs[0]), (self.temporal_patch_size, 1, 1, 1))

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

            # flatten_patches, (grid_t, grid_h, grid_w)
            thw_grids = [(grid_t, grid_h, grid_w)]

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
            image_thw_grids=thw_grids,
            video_thw_grids=None,
            image_input_mask=image_input_mask,
            video_input_mask=None,
            text=input_ids,
            target=target,
        )

    def batch(self, samples: List[ImageTaskSample]) -> VQATaskBatch:
        # Stack images to [num_tiles, c, h, w]. If there are no images (text-only), then use a dummy image.
        imgs = [s.imgs for s in samples]
        if len(imgs) > 0:
            imgs = torch.cat([torch.from_numpy(img) for img in imgs])
        else:
            imgs = torch.empty([0, 3 * self.temporal_patch_size * self.patch_size * self.patch_size], dtype=torch.float32)
        
        thw_grids = [thw_grids for s in samples for thw_grids in s.image_thw_grids]
        if len(thw_grids) > 0:
            thw_grids = torch.from_numpy(np.array(thw_grids)).long()
            assert thw_grids.prod(dim=-1).sum() == imgs.shape[0]
        else:
            thw_grids = torch.empty([0, 3], dtype=torch.long)
        
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
            image_thw_grids=thw_grids,
            video_thw_grids=None,
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
