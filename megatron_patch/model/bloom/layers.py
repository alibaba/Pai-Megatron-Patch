# Copyright (c) 2023 Alibaba PAI and Nvidia Megatron-LM Team.
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

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from megatron import get_args
from megatron.core import mpu
from megatron.core.parallel_state import get_tensor_model_parallel_rank
from megatron.core.parallel_state import get_tensor_model_parallel_world_size
from megatron.core.tensor_parallel.layers import _initialize_affine_weight_gpu
from megatron.core.tensor_parallel.layers import _initialize_affine_weight_cpu
from megatron.core.tensor_parallel.mappings import \
    reduce_from_tensor_model_parallel_region
from megatron.core.tensor_parallel.utils import VocabUtility


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.

    Keyword Arguments:
        init_method: method to initialize weights.
        params_dtype
        use_cpu_initialization
        perform_initialization
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 *,
                 init_method=init.xavier_normal_,
                 params_dtype: torch.dtype = torch.float32,
                 use_cpu_initialization: bool = False,
                 perform_initialization: bool = True):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size(
        )
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, get_tensor_model_parallel_rank(),
                self.tensor_model_parallel_size)
        self.num_embeddings_per_partition = self.vocab_end_index - \
            self.vocab_start_index
        args = get_args()
        if mpu.is_pipeline_first_stage() and args.embed_layernorm:
            from megatron.model.fused_layer_norm\
                import MixedFusedLayerNorm as LayerNorm
            self.norm = LayerNorm(embedding_dim)

        # Allocate weights and initialize.
        if use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(self.num_embeddings_per_partition,
                            self.embedding_dim,
                            dtype=params_dtype))
            if perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight,
                    self.num_embeddings,
                    self.embedding_dim,
                    self.num_embeddings_per_partition,
                    0,
                    init_method,
                    params_dtype=params_dtype)
        else:
            self.weight = Parameter(
                torch.empty(self.num_embeddings_per_partition,
                            self.embedding_dim,
                            device=torch.cuda.current_device(),
                            dtype=params_dtype))
            if perform_initialization:
                _initialize_affine_weight_gpu(self.weight,
                                              init_method,
                                              partition_dim=0,
                                              stride=1)

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | \
                         (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)

        if hasattr(self, 'norm'):
            output = self.norm(output)

        return output
