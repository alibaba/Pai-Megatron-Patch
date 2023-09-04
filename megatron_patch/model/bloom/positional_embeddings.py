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

import math

import torch


class SinusoidalPositionalEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half):
        super().__init__()
        inv_freq = 1.0 / (base**(torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.precision = precision

    def forward(self, x, seq_dim=1):
        t = torch.arange(x.shape[seq_dim],
                         device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum('i,j->ij', t, self.inv_freq)
        if self.precision == torch.bfloat16:
            sinusoid_inp = sinusoid_inp.float()
        sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos()
        if self.precision == torch.bfloat16:
            sin, cos = sin.bfloat16(), cos.bfloat16()
        emb = torch.cat((sin, cos), dim=-1)
        return emb[None, :, :]


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half):
        super().__init__()
        inv_freq = 1.0 / (base**(torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.precision = precision

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
            if self.precision == torch.bfloat16:
                self.cos_cached = self.cos_cached.bfloat16()
                self.sin_cached = self.sin_cached.bfloat16()
        return self.cos_cached, self.sin_cached


# rotary pos emb helpers:


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat(
        (-x2, x1),
        dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos, sin = (
        cos[offset:q.shape[0] + offset, ...],
        sin[offset:q.shape[0] + offset, ...],
    )
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) *
                                                            sin)


def apply_rotary_pos_emb_torch(q,
                               k,
                               cos,
                               sin,
                               offset: int = 0):  # jitting fails with bf16
    cos, sin = (
        cos[offset:q.shape[0] + offset, ...],
        sin[offset:q.shape[0] + offset, ...],
    )
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) *
                                                            sin)


class AliBi(torch.nn.Module):
    def __init__(self, num_heads, mp_size=1, mp_rank=1):
        super().__init__()
        # megatron splits across heads, so we need to make sure each
        # head receives the correct matrix
        assert mp_size <= num_heads and mp_rank <= mp_size
        self.mp_size = mp_size
        self.mp_rank = mp_rank
        self.num_heads = num_heads
        self.slice_size = num_heads // mp_size
        self.cached_matrix = None
        self.cached_seq_len = None
        slopes = torch.Tensor(
            self._get_slopes(num_heads))[mp_rank *
                                         self.slice_size:(mp_rank + 1) *
                                         self.slice_size]
        self.register_buffer('slopes', slopes)

    def _get_slopes(self, n):
        """
        Get slopes for Alibi positional embedding
        n : int = number of heads.
        For best performance, restrict n to a power of 2.
        """
        def get_slopes_power_of_2(n):
            start = 2**(-(2**-(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return (get_slopes_power_of_2(closest_power_of_2) +
                    self._get_slopes(
                        2 * closest_power_of_2)[0::2][:n - closest_power_of_2])

    def forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]

        # Initialize the AliBi matrix to match the
        # first provided key length; grow it exponentially
        # afterwards if longer inputs are provided.
        # This is important for inference, where we will
        # encounter progressively longer samples;
        # it should have no effect at training time.
        if self.cached_seq_len is not None and\
                self.cached_seq_len >= seq_len_k:
            a = self.cached_matrix
        else:
            target_seq_len = (seq_len_k if self.cached_seq_len is None else
                              self.cached_seq_len * 4)
            a = -torch.tril(
                torch.arange(target_seq_len).view(target_seq_len, 1).repeat(
                    1, target_seq_len) + torch.arange(0, -target_seq_len, -1))
            a = a.to(x.device).to(x.dtype)
            slopes = self.slopes.to(a.device).to(a.dtype)
            a = a * slopes.view(self.slopes.shape[0], 1, 1)
            self.cached_seq_len = target_seq_len
            self.cached_matrix = a

        # If the AliBi matrix is larger than the key length, clip it.
        if self.cached_seq_len > seq_len_k:
            a = self.cached_matrix[:, :seq_len_k, :seq_len_k]

        if seq_len_q != seq_len_k:
            # In the train case x has
            # dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens
            # is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not
            # equal to sk. sq only contains
            # one token (the last one in the full sequence)
            # In this case we use the appropriate
            # token index of the cache matrix.
            # As the cache matrix could already be bigger from a
            # past inference, not the last
            # token index in the sq sequence is used
            assert (
                seq_len_q == 1
            ), 'assumption sq == sk unless at inference' \
               ' time with cache in layer_past with sq == 1'
            a = a[:, seq_len_k - 1, :].view(
                a.shape[0], 1, a.shape[2])  # seq_len_k - 1 points to the last
            # token index in the current inference batch.

        return x + a
