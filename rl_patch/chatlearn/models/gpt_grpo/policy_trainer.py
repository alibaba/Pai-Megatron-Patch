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

from typing import Union
import torch
import os
from contextlib import nullcontext
import inspect

from megatron.core.transformer.spec_utils import import_module
from megatron.core.enums import ModelType
from megatron.training.training import save_checkpoint_and_time, print_datetime,  setup_model_and_optimizer
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.training import get_timers
from megatron.training.checkpointing import load_checkpoint
from megatron.training import get_model
from megatron.training import get_args, print_rank_0
from megatron.core import mpu
from megatron.training.utils import calc_params_l2_norm
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.training.utils import unwrap_model,logical_and_across_model_parallel_group
from megatron.core.utils import get_model_config
from megatron.core.distributed import finalize_model_grads
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)

import chatlearn
from chatlearn import MegatronModule

from tokenizer import build_tokenizer
from .policy_model import PolicyModel
from .train_helper import training_log, forward_step, inference_forward_step

REF_TAG = "ref_logprobs"
OLD_TAG = "old_logprobs"

class MegatronPolicyTrainer(MegatronModule):

    def setup(self):
        self.stats = {}
        self.buffer = {}
        self.args = get_args()
        self.report_memory_flag = True
        self.iteration_for_log = 0
        chatlearn_args = chatlearn.get_args()
        build_tokenizer(chatlearn_args)

        if self.trainable:
            self.model, self.optimizer, self.opt_param_scheduler = setup_model_and_optimizer(self.model_provider, ModelType.encoder_or_decoder)
            self.config = get_model_config(self.model[0])
            self.config.grad_scale_func = self.optimizer.scale_loss
            self.config.finalize_model_grads_func = finalize_model_grads

        else:
            self.model = get_model(self.model_provider, wrap_with_ddp=False)
            if self.args.load is not None:
                print(f"reference loading : {self.args.load}")
                _, _ = load_checkpoint(
                    self.model, None, None, checkpointing_context={},
                    skip_load_to_model_and_opt=False and getattr(self.args, "use_torch_fsdp2",
                                                                 False) and self.args.ckpt_format == "torch_dist")
            if int(os.environ.get("WORLD_SIZE", 1)) > 1:
                torch.distributed.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", 0))])

    def model_provider(self, pre_process=True, post_process=True) -> Union[PolicyModel]:
        args = get_args()
        use_te = args.transformer_impl == "transformer_engine"

        if args.record_memory_history:
            torch.cuda.memory._record_memory_history(True,
                                                     # keep 100,000 alloc/free events from before the snapshot
                                                     trace_alloc_max_entries=100000,

                                                     # record stack information for the trace events
                                                     trace_alloc_record_context=True)

            def oom_observer(device, alloc, device_alloc, device_free):
                # snapshot right after an OOM happened
                print('saving allocated state during OOM')
                snapshot = torch.cuda.memory._snapshot()
                from pickle import dump
                dump(snapshot, open(f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}", 'wb'))

            torch._C._cuda_attach_out_of_memory_observer(oom_observer)

        print_rank_0('building GPT model ...')
        # Experimental loading arguments from yaml
        if args.yaml_cfg is not None:
            config = core_transformer_config_from_yaml(args, "language_model")
        else:
            config = core_transformer_config_from_args(args)

        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_te,
                                                                    normalization=args.normalization)
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm)
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm,
                        normalization=args.normalization)
        mtp_block_spec = None
        if args.mtp_num_layers is not None:
            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, use_transformer_engine=use_te)

        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except:
                raise RuntimeError(
                    "--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found.")

        with build_model_context(**build_model_context_args):
            model = PolicyModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                parallel_output=True,
                share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                position_embedding_type=args.position_embedding_type,
                rotary_percent=args.rotary_percent,
                rotary_base=args.rotary_base,
                rope_scaling=args.use_rope_scaling,
                mtp_block_spec=mtp_block_spec,
            )

        return model

    def train_step(self, data_list):
        args = get_args()
        timers = get_timers()
        self.model[0].module.train()
        data_iterator = iter(data_list)

        self.optimizer.zero_grad()

        # Forward pass.
        timers('forward-backward', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        forward_backward_func = get_forward_backward_func()

        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=self.model,
            num_microbatches=get_num_microbatches(),
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False)

        timers('forward-backward').stop()

        # Empty unused memory.
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        # Update parameters.
        timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()
        timers('optimizer').stop()

        update_successful = logical_and_across_model_parallel_group(update_successful)

        # Update learning rate.
        if update_successful:
            increment = get_num_microbatches() * \
                        args.micro_batch_size * \
                        args.data_parallel_size
            self.opt_param_scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1

        # Empty unused memory.
        if args.empty_unused_memory_level >= 2:
            torch.cuda.empty_cache()

        loss_reduced = {}
        if mpu.is_pipeline_last_stage(ignore_virtual=True) or \
                args.model_type == ModelType.encoder_or_decoder_with_lbl:
            total_losses = {}
            for i in range(len(losses_reduced)):
                for key in losses_reduced[i].keys():
                    if key not in total_losses:
                        total_losses[key] = []
                    total_losses[key].append(losses_reduced[i][key])
            # Average loss across microbatches.

            # accumulate and average lbl_loss and z-loss for MoE
            # Get all keys; looking at first element in losses_reduced is insufficient with
            # virtual stages and models with LBL since only the last virtual stage in the
            # last physical stage has the true loss and the LBL, while all other stages have
            # LBL only.
            # different python processes may iterate over the same set in different orders, so list is used here.
            keys = sorted(list(total_losses.keys()))
            for key in keys:
                losses_reduced_for_key = torch.stack([item[0] for item in total_losses[key]])
                num_tokens_reduced_for_key = torch.stack([item[1] for item in total_losses[key]])
                loss_reduced[key] = losses_reduced_for_key.sum() / num_tokens_reduced_for_key.sum()
                # Load balancing losses need to be summed across virtual stages (not averaged),
                # so multiply back the number of virtual stages in a physical stage.
                if key == "load balancing loss":
                    if args.virtual_pipeline_model_parallel_size is not None:
                        loss_reduced[key] *= args.virtual_pipeline_model_parallel_size
                if key == "router z loss":
                    if args.virtual_pipeline_model_parallel_size is not None:
                        loss_reduced[key] *= args.virtual_pipeline_model_parallel_size

            #return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad, {}
        #return {}, skipped_iter, grad_norm, num_zeros_in_grad, {}
        self.iteration_for_log += 1


        self.args.consumed_train_samples += mpu.get_data_parallel_world_size() * \
                                                self.args.micro_batch_size * \
                                                get_num_microbatches()

        # Logging.
        loss_scale = self.optimizer.get_loss_scale().item()
        params_norm = None
        if self.args.log_params_norm:
            params_norm = calc_params_l2_norm(self.model)
        report_memory_flag = training_log(loss_reduced, {},
                                          self.optimizer.param_groups[0]['lr'],
                                          self.iteration_for_log, loss_scale,
                                          self.report_memory_flag, skipped_iter,
                                          grad_norm, params_norm, num_zeros_in_grad,
                                          self.stats, {}, "policy_trainer",
                                          self._metric_list)

        self.report_memory_flag = report_memory_flag

    def forward_step(self, data):
        args = get_args()

        for model_module in self.model:
            model_module.eval()

        output_log_probs = None
        with torch.no_grad():
            forward_backward_func = get_forward_backward_func()
            ref_nll = forward_backward_func(
                forward_step_func=inference_forward_step,
                data_iterator=data,
                model=self.model,
                num_microbatches=1,
                seq_length=args.seq_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.decoder_seq_length,
                forward_only=True,
                collect_non_loss_data=True)

            if mpu.is_pipeline_last_stage():
                output_log_probs = -ref_nll[0]

        # Move model back to the train mode.
        for model_module in self.model:
            model_module.train()

        if mpu.is_pipeline_last_stage():
            tag = OLD_TAG
            if OLD_TAG in data.keys():
                tag = REF_TAG
            data.update({tag: output_log_probs})
            return data

