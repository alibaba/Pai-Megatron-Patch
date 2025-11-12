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
"""Pretrain Qwen2.5-VL."""

import os
from functools import partial
from copy import deepcopy
from typing import Union, Optional, Tuple

import torch
import torch._dynamo
from megatron.core import mpu

from megatron.core import parallel_state
from megatron.training.checkpointing import get_checkpoint_name
from megatron.core.enums import ModelType
from megatron.training import get_args, get_timers, pretrain, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import (
    average_losses_across_data_parallel_group,
)
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron_patch.arguments import get_patch_args

from megatron_patch.model.qwen2_5_vl.layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_qwen2vl_vision_model_spec,
    get_mlp_module_spec

)
from megatron_patch.model.qwen2_5_vl.model import Qwen2_5VLModel

from megatron_patch.tokenizer import get_tokenizer
from megatron_patch.tensor_parallel import broadcast_data

torch._dynamo.config.suppress_errors = True
from megatron_patch.model.qwen2_5_vl.transformer_config import (
    Qwen2VLTransformerConfig,
    get_vision_model_config,
    get_vision_projection_config
)
from megatron.core import mpu, tensor_parallel
from megatron.core.parallel_state import get_tensor_model_parallel_rank
from megatron_patch.data.multimodal_dataset_helper import TaskEncoder, print_error_handler
from megatron.training.utils import unwrap_model

from megatron.energon import (
    LimitDataset,
    RepeatDataset,
    WorkerConfig,
    get_loader,
    get_savable_loader,
    get_train_dataset,
    get_val_datasets,
)



def model_provider(
    pre_process=True, post_process=True, add_encoder=True, add_decoder=True, vp_stage: Optional[int] = None
) -> Union[Qwen2_5VLModel]:
    args = get_args()
    print_rank_0("start building qwen2-vl model ...")

    # Config of vit, llm and projector
    config = core_transformer_config_from_args(args, Qwen2VLTransformerConfig)
    use_te = args.transformer_impl == "transformer_engine"
    if not use_te:
        raise NotImplementedError("The Qwen2-VL model is only implemented with TransformerEngine!")
    
    if args.rotary_seq_len_interpolation_factor is not None or args.rotary_seq_len_interpolation_factor != 1:
        print_rank_0('Multimodal RoPE currently not support RoPE interpolation, set to None...')
        args.rotary_seq_len_interpolation_factor = None

    vision_config = get_vision_model_config(args, deepcopy(config))
    vision_config.pipeline_model_parallel_size = 1
    vision_config.num_layers_in_first_pipeline_stage = None
    vision_projector_config = get_vision_projection_config(deepcopy(config), vision_config.hidden_size, vision_config.spatial_merge_size)
    
    print_rank_0("building Qwen2-5-VL model in TE...")
    # Layer Specs of vit, llm and projector
    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.qk_layernorm)
    vision_model_spec = get_qwen2vl_vision_model_spec()
    vision_projector_spec = get_mlp_module_spec(add_norm=False).submodules

    model = Qwen2_5VLModel(
        language_transformer_config=config,
        language_transformer_layer_spec=transformer_layer_spec,
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=args.max_position_embeddings,

        vision_transformer_config=vision_config,
        vision_transformer_layer_spec=vision_model_spec,
        drop_vision_class_token=False, # NOTE: no class token to drop?

        vision_projection_config=vision_projector_config,
        vision_projection_layer_spec=vision_projector_spec, 
        vision_projection_type='mlp',
        allow_missing_vision_projection_checkpoint= False, # TODO: may parameterized

        language_position_embedding_type=args.position_embedding_type,
        language_rotary_percent=args.rotary_percent,
        language_rotary_base=args.rotary_base,
        
        pre_process=pre_process,
        post_process=post_process,
        add_decoder=add_decoder,
        add_encoder=add_encoder,

        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        language_share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        vp_stage=vp_stage
    )

    model.freeze(
        freeze_language_model=getattr(args, 'freeze_LM', False), 
        freeze_vision_model=getattr(args, 'freeze_ViT', False), 
        freeze_vision_projection=False
    )

    return model

def get_ltor_masks_and_position_ids(
        model,
        input_ids, 
        image_thw_grids,
        video_thw_grids,
        target, 
        pad_token, 
        second_per_grid_ts,
        ignore_index=None
    ):
    """Build masks and position id for left to right model."""
    # Position ids. [3 X bs X seqlen]
    position_ids, _ = model.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_thw_grids,
        video_grid_thw=video_thw_grids,
        second_per_grid_ts=second_per_grid_ts,
        attention_mask=input_ids != pad_token
    )

    # Loss mask.
    loss_mask = torch.ones(target.size(), dtype=torch.float, device=input_ids.device)
    loss_mask[target == pad_token] = 0.0  # mask paddings
    if ignore_index is not None:
        loss_mask[target == ignore_index] = 0.0  # mask prompts

    # Attention mask.
    attention_mask = None

    return attention_mask, loss_mask, position_ids

def get_batch(model, data_iterator):
    """Generate a batch"""
    imgs = None
    tokens = None
    labels = None
    loss_mask = None
    attention_mask = None
    position_ids = None

    # Broadcast data.
    torch.cuda.nvtx.range_push("get_data")
    if data_iterator is not None and get_tensor_model_parallel_rank() == 0:
        data = next(data_iterator)
    else:
        data = None

    data_text =  broadcast_data(["text"], data, torch.int64)["text"]
    target =  broadcast_data(["target"], data, torch.int64)["target"]
    # shape: num_tiles x c x h x w
    imgs = broadcast_data(["imgs"], data, torch.float32)["imgs"]

    # shape: num_tiles x c x h x w
    videos = broadcast_data(["videos"], data, torch.float32)["videos"]
    # shape: n_image_samples
    image_thw_grids = broadcast_data(["image_thw_grids"], data, torch.long)["image_thw_grids"]
    # shape: n_video_samples
    video_thw_grids = broadcast_data(["video_thw_grids"], data, torch.long)["video_thw_grids"]
    # shape: n_video_samples
    second_per_grid_ts = broadcast_data(['second_per_grid_ts'], data, torch.float32)['second_per_grid_ts']


    image_input_mask = broadcast_data(["image_input_mask"], data, torch.bool)["image_input_mask"]
    video_input_mask = broadcast_data(["video_input_mask"], data, torch.bool)["video_input_mask"]
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("index tokens")
    tokenizer = get_tokenizer()

    tokens = data_text.long().contiguous()
    labels = target.contiguous()

    assert tokens.shape == labels.shape, f"tokens: {tokens.shape} != labels: {labels.shape}"
    torch.cuda.nvtx.range_pop()

    # NOTE: no sequence packing in LLM inputs
    torch.cuda.nvtx.range_push("get_ltor_masks_and_position_ids")
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        model, tokens, image_thw_grids, video_thw_grids, labels, tokenizer.pad_token_id, second_per_grid_ts
    )
    torch.cuda.nvtx.range_pop()

    return (
        tokens, 
        labels, 
        loss_mask, 
        attention_mask, 
        position_ids, 
        imgs, 
        videos,
        image_thw_grids, 
        video_thw_grids,
        image_input_mask, 
        video_input_mask
    )

def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()

    loss = torch.stack([torch.sum(losses.view(-1) * loss_mask), loss_mask.sum()])
    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan().any(), (
            f"Rank {global_rank}: found NaN in local forward loss calculation. "
            f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
        )

    averaged_loss = average_losses_across_data_parallel_group(loss)
    averaged_loss = averaged_loss[0] / averaged_loss[1]

    return loss[0] / loss[1] * args.context_parallel_size, {"lm loss": averaged_loss}

def forward_step(data_iterator, model: Qwen2_5VLModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    timers = get_timers()
    # Get the batch.
    timers("batch-generator", log_level=2).start()
    (
        tokens, 
        labels, 
        loss_mask, 
        attention_mask, 
        position_ids, 
        imgs, 
        videos,
        image_thw_grids, 
        video_thw_grids,
        image_input_mask, 
        video_input_mask
    ) = get_batch(unwrap_model(model), data_iterator)
    timers("batch-generator").stop()

    vision_data = torch.cat([imgs, videos], dim=0)
    vision_grid = torch.cat([image_thw_grids, video_thw_grids], dim=0)

    output_tensor = model(
        input_ids = tokens,
        position_ids = position_ids,
        vision_data = vision_data,
        vision_grid_thw =  vision_grid,
        video_start_index = image_input_mask.sum().cpu().item(),
        image_input_mask = image_input_mask,
        video_input_mask = video_input_mask,
        attention_mask = attention_mask,
        labels = labels
    )

    return output_tensor, partial(loss_func, loss_mask)

def run_online_eval(model):
    """Run an evaluation benchmark during training."""
    # Do nothing.
    return []

def write_online_eval_to_tensorboard(data, iteration, writer):
    """Write online evaluation data to Tensorboard."""
    if not writer:
        return

    for item in data:
        for k, v in item.items():
            writer.add_scalar(k, v, iteration)

def datasets_provider(worker_config=None):
    """Create multimodal train, validation and test datasets."""
    args = get_args()
    train_dataset = get_train_dataset(
        args.train_data_path[0],
        batch_size=args.micro_batch_size,
        task_encoder=TaskEncoder(),
        worker_config=worker_config,
        virtual_epoch_length=1000,
        max_samples_per_sequence=100,
        shuffle_buffer_size=100,
        handler=print_error_handler,
        image_decode="pil",
    )

    val_datasets = get_val_datasets(
        args.valid_data_path[0],
        batch_size=args.micro_batch_size,
        # This is the total number over all workers
        # limit=args.eval_iters * get_num_microbatches(),
        task_encoder=TaskEncoder(),
        worker_config=worker_config,
        handler=print_error_handler,
        image_decode="pil",
    )
    val_datasets_without_source_datasets = [
        # Limit the dataset to eval_iters * num_microbatches
        LimitDataset(
            # Repeat the inner dataset in case it's too short
            RepeatDataset(val_ds, worker_config=worker_config),
            length=args.eval_iters * get_num_microbatches(),
            worker_config=worker_config,
            reset_after_epoch=True,
        )
        for val_ds, _src_ds in val_datasets
    ]

    return train_dataset, val_datasets_without_source_datasets, None


def train_valid_test_dataloaders_provider(train_val_test_num_samples):
    """Build multimodal train, validation and test dataloaders."""
    if get_tensor_model_parallel_rank() != 0:
        return None, None, None

    args = get_args()

    worker_debug_path = None
    worker_log_level = 0

    rank = parallel_state.get_data_parallel_rank()
    world_size = parallel_state.get_data_parallel_world_size()
    data_parallel_group = parallel_state.get_data_parallel_group()

    worker_config = WorkerConfig(
        rank=rank,
        world_size=world_size,
        num_workers=args.num_workers,
        data_parallel_group=data_parallel_group,
        worker_debug_path=worker_debug_path,
        worker_log_level=worker_log_level,
    )
    train_ds, valid_ds1, test_ds = datasets_provider(worker_config)

    train_dataloader = get_savable_loader(train_ds, worker_config=worker_config)
    if args.load is not None:
        if getattr(args, "dataloader_save", None):
            dp_rank = parallel_state.get_data_parallel_rank()
            data_save_name = get_checkpoint_name(
                args.dataloader_save,
                args.iteration,
                basename=f"train_dataloader_dprank{dp_rank:03d}.pt",
            )
            if os.path.exists(data_save_name):
                try:
                    dataset_state_dict = torch.load(data_save_name, map_location="cpu")
                    train_dataloader.restore_state_rank(dataset_state_dict["dataloader_state_dict"])
                    print_rank_0(f"restored dataset state from {data_save_name}")
                except Exception as e:
                    print_rank_0("loading dataloader checkpoint failed. Skipping. " + str(e))

    valid_dataloader = [
        EnergonDataloader(get_loader(valid_ds, worker_config=worker_config))
        for valid_ds in valid_ds1
    ]
    test_dataloader = None # NOTE: no test

    return EnergonDataloader(train_dataloader), valid_dataloader, None


class EnergonDataloader:
    """A wrapper to use Megatron Energon dataloader with the Megatron-LM training loop."""
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self._iter = iter(cyclic_iter(dataloader))

    def __next__(self):
        return self._iter.__next__()

    def __iter__(self):
        return self._iter.__iter__()

    def save_state(self):
        return self._dataloader.save_state_rank()


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x

if __name__ == "__main__":
    train_valid_test_dataloaders_provider.is_distributed = True

    pretrain(
        train_valid_test_dataloaders_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'Qwen2VLTokenizer'},
        extra_args_provider=get_patch_args,
        process_non_loss_data_func=write_online_eval_to_tensorboard,
        non_loss_data_func=run_online_eval,
    )
