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

import importlib.util

import torch
from torch import einsum, nn

try:
    from einops import rearrange
except ImportError:
    rearrange = None
from torch import nn

__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']

apply_rotary_emb_func = None
rms_norm = None
class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        """
        Initializes the RotaryEmbedding module.

        Args:
            dim: The dimensionality of the embeddings.
            base: The base used for the frequency scaling.
        """
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        if importlib.util.find_spec("einops") is None:
            raise RuntimeError("einops is required for Rotary Embedding")

        self._rotary_pos_emb_cache = None
        self._seq_len_cached = 0
        self._ntk_alpha_cached = 1.0

    def update_rotary_pos_emb_cache(self, max_seq_len, offset=0, ntk_alpha=1.0):
        """
        Updates the cache of rotary positional embeddings based on the given parameters.

        Args:
            max_seq_len: The maximum sequence length for which to compute embeddings.
            offset: An optional offset for the embedding indices.
            ntk_alpha: An optional scaling factor for the base.
        """
        seqlen = max_seq_len + offset
        if seqlen > self._seq_len_cached or ntk_alpha != self._ntk_alpha_cached:
            base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))
            self.inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, self.dim, 2, device=self.inv_freq.device).float()
                    / self.dim
                )
            )
            self._seq_len_cached = max(2 * seqlen, 16)
            self._ntk_alpha_cached = ntk_alpha
            seq = torch.arange(self._seq_len_cached, device=self.inv_freq.device)
            freqs = torch.outer(seq.type_as(self.inv_freq), self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            from einops import rearrange

            self._rotary_pos_emb_cache = rearrange(emb, "n d -> 1 n 1 d")

    def forward(self, max_seq_len, offset=0, ntk_alpha=1.0):
        self.update_rotary_pos_emb_cache(max_seq_len, offset, ntk_alpha)
        return self._rotary_pos_emb_cache[:, offset : offset + max_seq_len]


def _rotate_half(x):
    """
    Performs a half rotation on the input tensor.

    Args:
        x: The input tensor to rotate.

    Returns:
        The rotated tensor.
    """
    from einops import rearrange

    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        t: The input tensor.
        freqs: The rotary positional embeddings.

    Returns:
        The tensor with rotary positional embeddings applied.
    """
    if apply_rotary_emb_func is not None and t.is_cuda:
        t_ = t.float()
        freqs = freqs.squeeze(0).squeeze(1)
        cos = freqs[:, : freqs.shape[-1] // 2].cos()
        sin = freqs[:, : freqs.shape[-1] // 2].sin()
        output = apply_rotary_emb_func(t_, cos, sin).type_as(t)
        return output
    else:
        rot_dim = freqs.shape[-1]
        t_, t_x = t[..., :rot_dim], t[..., rot_dim:]
        t_ = t_.float()
        t_x = t_x.float()
        t_ = (t_ * freqs.cos()) + (_rotate_half(t_) * freqs.sin())
        return torch.cat((t_, t_x), dim=-1).type_as(t)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if rms_norm is not None and x.is_cuda:
            return rms_norm(x, self.weight, self.eps)
        else:
            output = self._norm(x.float()).type_as(x)
            return output * self.weight
