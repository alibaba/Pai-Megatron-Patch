#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Alibaba PAI Team.
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

import os
import sys
import copy
import torch
import logging
import datasets
import argparse
import transformers
from transformers import (
    LlamaModel,
    LlamaForCausalLM,
    LlamaConfig,
    TrainingArguments,
    default_data_collator,
    Trainer, AutoTokenizer, AutoModelForCausalLM
)
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, LlamaDecoderLayer
from torch import nn
from datasets import load_dataset
from torch import cat, stack, arange, int32
from typing import Optional, Tuple
from einops import rearrange
from transformers.deepspeed import is_deepspeed_zero3_enabled, deepspeed_config
from transformers.utils import ContextManagers
from transformers.modeling_utils import no_init_weights
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import deepspeed

logger = logging.getLogger(__name__)
os.environ["WANDB_DISABLED"] = "true"


def dummy_function(*args, **kwargs):
    return None


try:
    from flash_attn.bert_padding import unpad_input, pad_input

    support_flash_attn, fa_version = True, 2.0
except ImportError:
    logger.info("flash attention unavailable")
    unpad_input = dummy_function
    pad_input = dummy_function
    flash_attn_varlen_qkvpacked_func = dummy_function
    support_flash_attn = False
if support_flash_attn:
    try:
        from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func

        flash_attn_varlen_qkvpacked_func = flash_attn_unpadded_qkvpacked_func
        fa_version = 1.0
    except ImportError:
        try:
            import flash_attn
            from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func, flash_attn_varlen_func, \
                flash_attn_varlen_kvpacked_func

            flash_attn.flash_attn_interface.flash_attn_unpadded_func = flash_attn_varlen_func
            flash_attn.flash_attn_interface.flash_attn_unpadded_qkvpacked_func = flash_attn_varlen_qkvpacked_func
            flash_attn.flash_attn_interface.flash_attn_unpadded_kvpacked_func = flash_attn_varlen_kvpacked_func
            fa_version = 2.0
        except ImportError:
            flash_attn_varlen_qkvpacked_func = dummy_function


def get_patch_args(parser):
    group = parser.add_argument_group(title='llama2')
    group.add_argument('--local-rank', type=int, default=None, help='local rank passed from distributed launcher')
    group.add_argument('--micro-batch-size', type=int, default=None,
                       help='Batch size per model instance (local batch size). Global batch size is local batch size '
                            'times data parallel size times number of micro batches.')
    group.add_argument('--gradient-accumulation-steps', type=int, default=None)
    group.add_argument('--epochs', type=int, default=None, help='Number of finetunning epochs. '
                                                                'Zero results in evaluation only.')
    group.add_argument('--num-layers', type=int, default=None, help='Number of transformer layers.')
    group.add_argument('--hidden-size', type=int, default=None, help='Tansformer hidden size.')
    group.add_argument('--intermediate-size', type=int, default=None, help='--intermediate-size')
    group.add_argument('--num-attention-heads', type=int, default=None, help='Number of transformer attention heads.')
    group.add_argument('--seq-length', type=int, default=None, help='Maximum sequence length to process.')
    group.add_argument('--num-workers', type=int, default=None)
    group.add_argument('--logging-dir', type=str, default='megatron', help='transformer-type')
    group.add_argument('--train-data', default=None, help='Whitespace separated paths or corpora names for training.')
    group.add_argument('--valid-data', default=None, help='path(s) to the validation data.')
    group.add_argument('--enable-gradient-checkpointing', action='store_true')
    group.add_argument('--fp16', action='store_true', help='Run model in fp16 mode.')
    group.add_argument('--bf16', action='store_true', help='Run model in bfloat16 mode.')
    group.add_argument('--load', type=str, default=None)
    group.add_argument('--save', type=str, default=None)
    group.add_argument('--lr', type=float, default=None,
                       help='Initial learning rate. Depending on decay style and initial warmup, the learing rate at '
                            'each iteration would be different.')
    parser.add_argument('--model', type=str, help='name and size of the model',
                        choices=['llama2-13b', 'qwen-7b', 'qwen-14b', 'qwen1.5-32b', 'qwen-72b'])
    parser.add_argument('--flash', action='store_true', help='use flash attention, only work for llama2-13b.')
    return parser


parser = argparse.ArgumentParser(description='PyTorch LLaMA Training')
parser = get_patch_args(parser)
args = parser.parse_args()


class LlamaAttentionWithFlash(LlamaAttention):
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        kv_seq_len = key_states.shape[-2]
        assert past_key_value is None, "past_key_value is not supported"
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        assert not output_attentions, "output_attentions is not supported"
        if past_key_value is not None:
            key_states = cat([past_key_value[0], key_states], dim=2)
            value_states = cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None
        qkv = stack([query_states, key_states, value_states], dim=2)
        qkv = qkv.transpose(1, 3)
        key_padding_mask = attention_mask
        if key_padding_mask is None:
            qkv = rearrange(qkv, "b s ... -> (b s) ...")
            max_s = q_len
            cu_q_lens = arange(0, (bsz + 1) * q_len, step=q_len, dtype=int32, device=qkv.device)
            output = flash_attn_varlen_qkvpacked_func(qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True)
            output = rearrange(output, "(b s) ... -> b s ...", b=bsz)
        else:
            nheads = qkv.shape[-2]
            x = rearrange(qkv, "b s three h d -> b s (three h d)")
            x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
            x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
            output_unpad = flash_attn_varlen_qkvpacked_func(x_unpad, cu_q_lens, max_s, 0.0,
                                                            softmax_scale=None, causal=True)
            output = rearrange(
                pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, q_len),
                "b s (h d) -> b s h d",
                h=nheads,
            )
        return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value


class LlamaDecoderLayerWithFlash(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.self_attn = LlamaAttentionWithFlash(config=config)


class LlamaModelWithFlash(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        # replace by layer with flash attention
        self.layers = nn.ModuleList([LlamaDecoderLayerWithFlash(config) for _ in range(config.num_hidden_layers)])
        self.post_init()

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        return attention_mask


class LlamaForCausalLMWithFlash(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModelWithFlash(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # initialize weights again and apply final processing after the replacement
        self.post_init()


def tokenize(strings, tokenizer):
    tokenized_list = [
        tokenizer(
            text,
            return_tensors='np',
            padding='max_length',
            max_length=args.seq_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        (tokenized.input_ids != tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(sources, targets, tokenizer):
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [tokenize(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized['input_ids']
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized['input_ids_lens']):
        label[:source_len] = -100
    return dict(input_ids=input_ids, labels=labels)


def init_weight(model):
    def _init_weight(module):
        if hasattr(module, 'weight'):
            nn.init.normal_(module.weight, 0., 0.1)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, 0)

    model.apply(_init_weight)


def save_hf_format(model, tokenizer, args, sub_folder=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(args.save, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    for key in list(save_dict.keys()):
        if "lora" in key:
            del save_dict[key]
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)

    model_to_save = model_ema.module if hasattr(model_ema, 'module') else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v]), enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict


def main():
    training_args = TrainingArguments(
        output_dir=args.save,
        overwrite_output_dir=True,
        bf16=args.bf16,
        fp16=args.fp16,
        deepspeed='./ds_config.json',
        do_train=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        do_eval=False,
        per_device_eval_batch_size=args.micro_batch_size,
        dataloader_num_workers=args.num_workers,
        learning_rate=args.lr,
        adam_beta1=0.9,
        adam_beta2=0.95,
        warmup_ratio=0.1,
        logging_strategy='steps',
        log_level='info',
        logging_dir=args.logging_dir,
        logging_steps=1,
        disable_tqdm=False,
        save_strategy='epoch',
        ddp_timeout=18000
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    transformers.set_seed(training_args.seed)

    data_files = {
        "train": [os.path.join(args.train_data, f) for f in os.listdir(args.train_data)] if os.path.isdir(
            args.train_data) else args.train_data,
        "validation": [os.path.join(args.valid_data, f) for f in os.listdir(args.valid_data)] if os.path.isdir(
            args.valid_data) else args.valid_data
    }

    raw_datasets = load_dataset(
        'json',
        data_files=data_files,
        cache_dir=None,
        use_auth_token=None,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.load, trust_remote_code=True)
    if args.model in ('qwen-7b', 'qwen-14b', 'qwen1.5-32b', 'qwen-72b'):
        tokenizer.eos_token = '<|endoftext|>'
        tokenizer.pad_token = tokenizer.eos_token

    init_contexts = [no_init_weights(_enable=False)]
    if is_deepspeed_zero3_enabled():
        logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
        init_contexts = [deepspeed.zero.Init(config_dict_or_path=deepspeed_config())] + init_contexts
    if args.model == 'llama2-13b':
        config = transformers.CONFIG_MAPPING['llama'](
            num_hidden_layers=args.num_layers,
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.intermediate_size,
            use_cache=True
        )
        if args.flash:
            if not support_flash_attn:
                raise ImportError('FlashAttention is not available, please install with pip install flash-attn')
            logger.info(f"enable flash attention {fa_version}")
            if args.load:
                # init_contexts will be constructed in the `from_pretrained`
                model = LlamaForCausalLMWithFlash.from_pretrained(
                    args.load,
                    from_tf=False,
                    config=config,
                    revision='main',
                    use_auth_token=None,
                    low_cpu_mem_usage=False,
                )
            else:
                with ContextManagers(init_contexts):
                    model = LlamaForCausalLMWithFlash(config)
                    init_weight(model)
        else:
            logger.info("disable flash attention")
            if args.load:
                model = LlamaForCausalLM.from_pretrained(
                    args.load,
                    from_tf=False,
                    config=config,
                    revision='main',
                    use_auth_token=None,
                    low_cpu_mem_usage=False,
                )
            else:
                with ContextManagers(init_contexts):
                    model = LlamaForCausalLM(config)
                    init_weight(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.load,
            from_tf=False,
            torch_dtype=torch.float16 if args.fp16 else torch.bfloat16,
            trust_remote_code=True
        )

    model.to('cuda')
    if args.enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    n_params = model.num_parameters()
    logger.info(f"Training model with {n_params * 1e-9:.2f}B model")
    embedding_size = model.get_input_embeddings().weight.shape[0]

    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    def tokenize_function(examples):
        sources = examples['input']
        targets = [x + tokenizer.eos_token for x in examples['output']]
        data_dict = preprocess(sources, targets, tokenizer)
        return data_dict

    with training_args.main_process_first(desc="dataset map tokenization"):
        num_proc = 1 if args.model in ('qwen-7b', 'qwen-14b', 'qwen1.5-32b', 'qwen-72b') else 64
        lm_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=num_proc
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    trainer.train()

    if args.save is not None:
        args.global_rank = torch.distributed.get_rank()
        print('saving the final model ...', args.global_rank)

        if args.global_rank == 0:
            save_hf_format(model, tokenizer, args)

        if is_deepspeed_zero3_enabled():
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(model, args.global_rank, args.save, zero_stage=3)


if __name__ == "__main__":
    main()
