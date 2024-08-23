import torch
from torch import nn

class LLamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, config=None):
        """
        LLamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # NOTE: In Transformer Engine, the following line in RMSNorm is implemented with 
        # (self.weight.float() * hidden_states).to(input_dtype)
        return self.weight * hidden_states.to(input_dtype)

# TERMSNorm is equivalent to the following code:
# class TorchRMSNorm(nn.Module):
#     def __init__(self, in_features, eps=1e-5):
#         super().__init__()

#         self.eps = eps
#         self.in_features = in_features

#         self.weight = nn.Parameter(torch.ones(in_features))
#         self.register_parameter("weight", self.weight)

#     def forward(self, x):
#         norm_x2 = torch.sum(x.float()**2, dim=-1, keepdim=True)
#         d_x = self.in_features

#         rms_x2 = norm_x2 / d_x + self.eps
#         r_rms_x = rms_x2 ** (-1. / 2)
#         x_normed = x * r_rms_x

#         return (self.weight.float() * x_normed).to(x.dtype)