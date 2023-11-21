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

import sys
import argparse
import dataclasses
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


_GLOBAL_MG_V = None



def get_tasks_args(parser):
    group = parser.add_argument_group(title='patch')

    for action in vars(group)['_actions']:
        if isinstance(action, argparse._StoreAction):
            if '--tokenizer-type' in action.option_strings:
                action.default = "NullTokenizer"

    for action in vars(group)['_actions']:
        if isinstance(action, argparse._StoreAction):
            if '--vocab-size' in action.option_strings:
                action.default = -1

    group.add_argument('--fp8-format', default=None,
                       choices=['e4m3', 'hybrid'],
                       help='Which fp8 format scheme to use for FP8 tensors in the forward and backward pass',
                       dest='fp8')

    group.add_argument('--position-embedding-type', type=str, default='learned_absolute',
                       choices=['learned_absolute', 'rope'],
                       help='Position embedding type.')

    group.add_argument('--normalization', default='LayerNorm',
                       choices=['LayerNorm', 'RMSNorm'],
                       help='Which normalization technique to use.')

    group.add_argument('--overlap-p2p-communication',
                       action='store_true',
                       help='overlap pipeline parallel communication with forward and backward chunks',
                       dest='overlap_p2p_comm')

    group.add_argument('--group-query-attention', action='store_true',
                       help='Use group-query attention.')

    group.add_argument('--num-query-groups', type=int, default=1)

    for action in vars(group)['_actions']:
        if isinstance(action, argparse._StoreAction):
            if '--recompute-num-layers' in action.option_strings:
                action.default = None

    group.add_argument('--rotary-seq-len-interpolation-factor', type=int, default=None,
                       help='Sequence length interpolation factor for rotary embeddings.')

    group.add_argument('--overlap-grad-reduce', action='store_true',
                       default=False, help='If set, overlap DDP grad reduce.')

    group.add_argument('--skip-train', action='store_true',
                       default=False, help='If set, bypass the training loop, '
                                           'optionally do evaluation for validation/test, and exit.')

    group.add_argument('--profile', action='store_true',
                       help='Enable nsys profiling. When using this option, nsys '
                            'options should be specified in commandline. An example '
                            'nsys commandline is `nsys profile -s none -t nvtx,cuda '
                            '-o <path/to/output_file> --force-overwrite true '
                            '--capture-range=cudaProfilerApi '
                            '--capture-range-end=stop`.')


    for action in vars(group)['_actions']:
        if isinstance(action, argparse._StoreAction):
            if '--position-embedding-type' in action.option_strings:
                action.choices.append('none')

    group.add_argument('--local-rank',
                       type=int,
                       default=None,
                       help='local rank passed from distributed launcher')

    group.add_argument('--n-head-kv',
                       type=int,
                       default=None,
                       help='n-head-kv')

    group.add_argument('--transformer-type',
                       type=str,
                       default='megatron',
                       help='transformer-type')

    group.add_argument('--max-padding-length',
                       type=int,
                       default=None,
                       help='max-padding-length')

    group.add_argument('--dataset',
                       type=str,
                       default=None,
                       help='dataset')

    group.add_argument('--pretrained-checkpoint',
                       type=str,
                       default=None,
                       help='Pretrained checkpoint used for finetunning.')

    group.add_argument('--epochs',
                       type=int,
                       default=None,
                       help='Number of finetunning epochs. Zero results in '
                       'evaluation only.')

    group.add_argument('--intermediate-size',
                       type=int,
                       default=None,
                       help='--intermediate-size')

    group.add_argument('--extra-vocab-size',
                       type=int,
                       default=1,
                       help='--extra-vocab-size')

    group.add_argument('--keep-last',
                       action='store_true',
                       help='Keep the last batch (maybe incomplete) in'
                       'the data loader')

    group.add_argument('--data-dir',
                       default=None,
                       help='data-dir')

    group.add_argument('--train-data',
                       nargs='+',
                       default=None,
                       help='Whitespace separated paths or corpora names '
                       'for training.')

    group.add_argument('--valid-data',
                       nargs='+',
                       default=None,
                       help='path(s) to the validation data.')

    group.add_argument('--patch-tokenizer-type',
                       type=str,
                       help='patch-tokenizer-type')

    group.add_argument('--use-alibi-mask',
                       action='store_true',
                       help='use alibi mask for baichuan model')

    group.add_argument('--use-normhead',
                       action='store_true',
                       help='use-normhead')

    group.add_argument('--glu-activation',
                       type=str,
                       help='GLU activations to use.')

    group.add_argument('--attention-head-type',
                       type=str,
                       default=None,
                       choices=['multihead', 'multiquery'],
                       help='Type of attention heads. `multihead` is the standard multi-head attention.'
                       '`multiquery` shares the values and keys across attention heads')

    group.add_argument('--transformer-timers',
                       action='store_true',
                       help="If set, activate the timers within the transformer layers."
                        "Only for debugging, as this slows down the model.")

    group.add_argument('--text-generate-input-file',
                       type=str,
                       default='')

    group.add_argument('--text-generate-output-file',
                       type=str,
                       default='')

    group.add_argument('--text-generate-gt-file',
                       type=str,
                       default='')

    group.add_argument('--time',
                       action='store_true',
                       help='measure end to end text generation average time')

    group.add_argument('--eval-dev',
                       action='store_true')

    group.add_argument('--input-len',
                       type=int,
                       default=1,
                       help='input lenth for measure end to end text generation average time')

    group.add_argument('--generation-length',
                       type=int,
                       default=None,
                       help='generation-seq-len')

    group.add_argument('--top-p',
                       type=float,
                       default=0.0,
                       help='Top p sampling.')

    group.add_argument('--top-k',
                       type=int,
                       default=0,
                       help='Top k sampling.')

    group.add_argument('--out-seq-length',
                       type=int,
                       default=1024,
                       help='Size of the output generated text.')

    group.add_argument('--temperature',
                       type=float,
                       default=1.0,
                       help='Sampling temperature.')

    group.add_argument('--repetition_penalty',
                       type=float,
                       default=1.1,
                       help='Repetition_penalty.')

    group.add_argument('--embed-layernorm',
                       action='store_true',
                       help='use layernorm for embedding')

    group.add_argument('--repetition-penalty',
                       type=float,
                       default=1.2,
                       help='Repetition_penalty.')


    group.add_argument('--source-seq-len',
                       type=int,
                       default=None,
                       help='source-seq-len')

    group.add_argument('--target-seq-len',
                       type=int,
                       default=None,
                       help='target-seq-len')

    group.add_argument('--position-encoding-2d',
                       action='store_true',
                       help='position-encoding-2d')

    group.add_argument('--z-loss-weight',
                       type=float,
                       default=0.0,
                       help='the max-z weight for baichuan2')


    group.add_argument('--use-llama2-rotary-position-embeddings', action='store_true',
                       help='Use llama2 rotary positional embeddings or not. '
                       'Deprecated: use --position-embedding-type')

    group.add_argument('--use-mistral-rotary-position-embeddings', action='store_true',
                       help='Use llama2 rotary positional embeddings or not. '
                       'Deprecated: use --position-embedding-type')

    group.add_argument('--mm-use-im-start-end',
                       action='store_true')

    group.add_argument('--mm-use-im-patch-token',
                       action='store_true')

    group.add_argument('--tune-mm-mlp-adapter',
                       action='store_true')

    group.add_argument('--image-folder',
                       type=str,
                       default='')

    group.add_argument('--mm-vision-select-layer',
                       type=int,
                       default=None)

    group.add_argument('--vision-tower',
                       type=str,
                       default='')

    group.add_argument('--image-aspect-ratio',
                       type=str,
                       default='square')

    group.add_argument('--version',
                       type=str,
                       default='plain')

    group.add_argument('--mm-projector-type',
                       type=str,
                       default=None)

    group.add_argument('--image-size',
                       type=int,
                       default=None,
                       help='image-size')

    group.add_argument('--patch-size',
                       type=int,
                       default=None,
                       help='patch-size')

    group.add_argument('--sliding-window', type=int, default=None)

    return parser





@dataclass
class ModelParallelConfig:
    """Base configuration for Megatron Core

    Model Parallelism
    -----------------

    tensor_model_parallel_size (int): Intra-layer model parallelism. Splits tensors across GPU ranks. Defaults to 1.

    pipeline_model_parallel_size (int): Inter-layer model parallelism. Splits transformer layers across GPU
        ranks. Defaults to 1.

    virtual_pipeline_model_parallel_size (int): Interleaved pipeline parallelism is used to improve performance by
        reducing the pipeline bubble.  Considers a transformer block as a list of smaller transformer (virtual) blocks.
        The number of virtual blocks per pipeline model parallel rank is the virtual model parallel size.  See Efficient
        Large-Scale Language Model Training on GPU Clusters Using Megatron-LM: https://arxiv.org/pdf/2104.04473.pdf for
        more details.  Defaults to None.

    sequence_parallel (bool): Makes tensor parallelism more memory efficient for LLMs (20B+) by
        parallelizing layer norms and dropout sequentially.  See Reducing Activation Recomputation in Large Transformer
        Models: https://arxiv.org/abs/2205.05198 for more details. Defaults to False.

    Initialization
    --------------

    perform_initialization (bool, default=True): If true, weights are initialized. This option can be useful when you
        know you are going to load values from a checkpoint.

    use_cpu_initialization: (bool, default=False): When set to False, we initialize the weights directly on the GPU.
        Transferring weights from CPU to GPU can take a significant amount of time for large models. Defaults to False.

    Training
    --------

    fp16 (bool): If true, train with fp16 mixed precision training. Defaults to False.

    bf16 (bool): If true, train with bf16 mixed precision training. Defaults to False.

    params_dtype (torch.dtype): dtype used when intializing the weights. Defaults to torch.float32

    timers (optional, default=None): TODO

    Optimizations
    -------------

    gradient_accumulation_fusion (bool): If true, fuses weight gradient accumulation to GEMMs. Requires the custom CUDA
        extension fused_weight_gradient_mlp_cuda module. To use gradient_accumulation_fusion you must install APEX with
        --cpp_ext and --cuda_ext. For example: "pip install --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext\"
        ". Note that the extension requires CUDA>=11. Otherwise, you must turn off gradient accumulation fusion.
        Defaults to False.

    async_tensor_model_parallel_allreduce (bool, default=True): If true, enables asynchronous execution of
        tensor-model-parallel all-reduce with weight gradient compuation of a column-linear layer.  Defaults to False.

    Pipeline Parallelism
    --------------------

    pipeline_dtype (required): dtype used in p2p communication, usually params_dtype

    grad_scale_func (optional, default=None): If using loss scaling, this function should take the loss and return the
        scaled loss. If None, no function is called on the loss.

    enable_autocast (bool): If true runs the forward step function inside torch.autocast context. Default is False.

    autocast_dtype (torch.dtype): dtype to pass to torch.amp.autocast when enabled. Default is pipeline_dtype.

    variable_seq_lengths (bool, default=False): Support for variable sequence lengths across microbatches. Setting this
        communicates the size of tensors during pipeline parallelism communication, because of this extra overhead it
        should only be set if the sequence length varies by microbatch within a global batch.

    num_microbatches_with_partial_activation_checkpoints (int, default=None): If int, set the number of microbatches
        where not all of the layers will be checkpointed and recomputed. The rest of the microbatches within the window
        of maximum outstanding microbatches will recompute all layers (either full recompute or selective recompute). If
        None, the checkpoint and recompute will be left up to the forward_step function.

    overlap_p2p_comm (bool, optional, default=False): When True some of the peer to peer communication for pipeline
        parallelism will overlap with computation. Must be False if batch_p2p_comm is true.

    batch_p2p_comm (bool, default=True): Use batch_isend_irecv instead of individual isend/irecv calls. Must be False
        if overlap_p2p_comm is True.

    batch_p2p_sync (bool, default=True): When using batch_isend_irecv, do a cuda.device.synchronize afterward to work
        around a bug in older version of PyTorch.

    use_ring_exchange_p2p (bool, default = False): Use custom ring_exchange kernel instead of
        torch.distributed.batch_isend_irecv(). Requires custom built torch with torch.distributed.ring_exchange.

    deallocate_pipeline_outputs (optional, default=False): If True, output data is deallocated after the tensor is sent
        to the next pipeline stage.  Helps with saving memory, does nothing when pipeline parallel is not used.

    no_sync_func (optional): Function that creates a context that suppresses asynchronous data-parallel
        communication. If the model is an instance of torch.nn.DistributedDataParallel, the default is to use
        torch.nn.DistributedDataParallel.no_sync.

    grad_sync_func (optional): Function that launches asynchronous gradient reductions (e.g. distributed optimizer
        gradient reduce-scatters). The function should take one argument: an iterable of parameters whose gradients are
        to be synchronized.

    param_sync_func (optional): Function that launches asynchronous parameter synchronizations (e.g. distributed
        optimizer parameter all-gathers). The function should take one argument: an iterable of parameters to be
        synchronized.

    """

    # Model parallelism
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: int = None
    sequence_parallel: bool = False

    # Initialization
    perform_initialization: bool = True
    use_cpu_initialization: bool = False

    # Training
    fp16: bool = False
    bf16: bool = False
    params_dtype: torch.dtype = torch.float32
    timers: Callable = None

    # Optimizations
    gradient_accumulation_fusion: bool = False
    async_tensor_model_parallel_allreduce: bool = False

    # Pipeline Parallel
    pipeline_dtype: torch.dtype = None
    grad_scale_func: Callable = None
    enable_autocast: bool = False
    autocast_dtype: torch.dtype = None
    variable_seq_lengths: bool = False
    num_microbatches_with_partial_activation_checkpoints: int = None
    overlap_p2p_comm: bool = False
    batch_p2p_comm: bool = True
    batch_p2p_sync: bool = True
    use_ring_exchange_p2p: bool = False
    deallocate_pipeline_outputs: bool = False
    no_sync_func: Callable = None
    grad_sync_func: Callable = None
    param_sync_func: Callable = None

    def __post_init__(self):
        """ Python dataclass method that is used to modify attributes after initialization.
            See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more details.
        """
        if self.sequence_parallel:
            if self.tensor_model_parallel_size <= 1:
                raise ValueError("Can not use sequence paralllelism without tensor parallelism")
            if self.async_tensor_model_parallel_allreduce:
                # sequence_parallelism already does this async
                self.async_tensor_model_parallel_allreduce = False

        if self.pipeline_model_parallel_size > 1:
            if self.pipeline_dtype is None:
                raise ValueError(
                    "When using pipeline parallelism, pipeline_dtype must be specified"
                )

        if self.autocast_dtype is None:
            self.autocast_dtype = self.params_dtype


@dataclass
class TransformerConfig(ModelParallelConfig):

    # model architecture
    num_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_query_groups: int = None

    ffn_hidden_size: int = None
    kv_channels: int = None
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    fp32_residual_connection: bool = False
    # @jcasper should we keep this option?
    apply_residual_connection_post_layernorm: bool = False
    layernorm_epsilon: float = 1e-5
    layernorm_zero_centered_gamma: bool = False
    add_bias_linear: bool = True
    gated_linear_unit: bool = False
    activation_func: Callable = F.gelu
    num_moe_experts: int = None

    # initialization
    init_method: Callable = None
    output_layer_init_method: Callable = None
    init_method_std: float = 0.02

    # mixed-precision
    apply_query_key_layer_scaling: bool = True
    attention_softmax_in_fp32: bool = True

    # communication

    # fusion
    bias_gelu_fusion: bool = False  # TODO: this should be bias_activation_fusion ?
    masked_softmax_fusion: bool = False
    persist_layer_norm: bool = False
    bias_dropout_fusion: bool = False  # TODO: this should be bias_dropout_add_fusion?

    # activation recomputation
    recompute_granularity: str = None
    recompute_method: str = None
    recompute_num_layers: int = None
    distribute_saved_activations: bool = None

    # fp8 related
    fp8: str = None
    fp8_margin: int = 0
    fp8_interval: int = 1
    fp8_amax_history_len: int = 1
    fp8_amax_compute_algo: str = "most_recent"
    fp8_wgrad: bool = True
    fp8_e4m3: bool = True

    # experimental section (TODO: move to apt. section above once stable)
    normalization: bool = "LayerNorm"  # alt value supported by TE: "RMSNorm"

    def __post_init__(self):
        """ Python dataclass method that is used to modify attributes after initialization.
            See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more details.
        """
        super().__post_init__()
        if self.fp16 and self.bf16:
            raise ValueError(
                f'Only one of self.fp16: {self.fp16} and self.bf16 {self.bf16} should be True.'
            )

        if self.num_attention_heads % self.tensor_model_parallel_size != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be a multiple of "
                f"tensor_model_parallel_size ({self.tensor_model_parallel_size})."
            )

        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = 4 * self.hidden_size

        if self.kv_channels is None:
            self.kv_channels = self.hidden_size // self.num_attention_heads

        if self.num_query_groups is None:
            self.num_query_groups = self.num_attention_heads

        if self.num_query_groups % self.tensor_model_parallel_size != 0:
            raise ValueError(
                f"num_query_groups ({self.num_query_groups}) must be a multiple of "
                f"tensor_model_parallel_size ({self.tensor_model_parallel_size})."
            )

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.recompute_granularity is not None:
            if not self.recompute_granularity in ['full', 'selective']:
                raise ValueError(
                    f'When using recompute_granuarlity: {self.recompute_granularity} must be "full" or "selective".'
                )

            if self.recompute_method is not None:
                if not self.recompute_method in ['block', 'uniform']:
                    raise ValueError(
                        f'recompute_method: {self.recompute_method} must be "block" or "uniform".'
                    )
            elif self.recompute_granularity != 'selective':
                raise ValueError(
                    f'Using recompute_granularity: {self.recompute_granularity} so recompute_method must be "block" or "uniform"'
                )

            if self.recompute_granularity != 'selective' and self.recompute_num_layers is None:
                raise ValueError(
                    f'When using recompute_granularity: {self.recompute_granularity} recompute_num_layers must be between '
                    f'1 and num_layers_per_pipeline_rank: {self.num_layers // self.pipeline_model_parallel_size}'
                )
            elif (
                self.recompute_granularity == 'selective' and self.recompute_num_layers is not None
            ):
                raise ValueError(
                    f'When using recompute_granularity: {self.recompute_granularity} recompute_num_layers must be None.'
                )

            if self.distribute_saved_activations and self.sequence_parallel:
                raise ValueError(
                    f'distribute_saved_activations: {self.distribute_saved_activations} must be false when sequence parallel is enabled: {self.sequence_parallel}'
                )

            if self.virtual_pipeline_model_parallel_size is not None:
                if not self.num_layers % self.virtual_pipeline_model_parallel_size == 0:
                    raise ValueError(
                        f'num_layers: {self.num_layers} must be divisible by virtual_model_parallel_size {self.virtual_pipeline_model_parallel_size}'
                    )

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.bias_gelu_fusion:
            if not self.add_bias_linear:
                raise ValueError(
                    "When bias_gelu_fusion is True, add_bias_linear must also be True."
                )

            if self.activation_func != F.gelu:
                raise ValueError(f'When bias_gelu_fusion is True, activation_func must be F.gelu.')

        from megatron.model.utils import init_method_normal, scaled_init_method_normal

        if self.init_method is None:
            self.init_method = init_method_normal(self.init_method_std)

        if self.output_layer_init_method is None:
            self.output_layer_init_method = scaled_init_method_normal(
                self.init_method_std, self.num_layers
            )


def core_transformer_config_from_args(args):

    # Translate args to core transformer configuration
    kw_args = {}
    for f in dataclasses.fields(TransformerConfig):
        if hasattr(args, f.name):
            kw_args[f.name] = getattr(args, f.name)
    kw_args['persist_layer_norm'] = not args.no_persist_layer_norm
    kw_args['layernorm_zero_centered_gamma'] = args.apply_layernorm_1p
    kw_args['deallocate_pipeline_outputs'] = True
    kw_args['pipeline_dtype'] = args.params_dtype
    kw_args['batch_p2p_comm'] = not args.overlap_p2p_comm
    if args.swiglu:
        kw_args['activation_func'] = F.silu
        kw_args['gated_linear_unit'] = True
        kw_args['bias_gelu_fusion'] = False
    if args.init_method_xavier_uniform:
        kw_args['init_method'] = torch.nn.init.xavier_uniform_
        kw_args['scaled_init_method'] = torch.nn.init.xavier_uniform_

    if args.group_query_attention:
        kw_args['num_query_groups'] = args.num_query_groups
    else:
        kw_args['num_query_groups'] = None

    return TransformerConfig(**kw_args)