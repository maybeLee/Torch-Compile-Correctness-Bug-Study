"""
Bug report for identifier: transpose + max_pool
Buggy Version: torch 2.12.0
"""

import torch
import torch._inductor.config
torch._inductor.config.fallback_random = True

import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        out = F.adaptive_max_pool2d(x, (2, 2), return_indices=True)
        return out[1]

model = Model()
cmodel = torch.compile(model)
torch.manual_seed(0)
x = torch.randn(2, 4, 12, 12).transpose(2, 3)
out1 = model(x)
out2 = cmodel(x)
torch.testing.assert_close(out1, out2)
