"""
Bug report for identifier: arange %2 -> duplicate indices
Buggy Version: torch 2.12.0
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        d = x + 0
        v = x.view(-1)
        idx = torch.arange(v.numel()) % 2
        v[idx] = v[idx] + 1.0
        return d[:, :2] + x

model = Model()
cmodel = torch.compile(model)

x1 = torch.tensor([[1.,2.], [3.,4.]])
x2 = torch.tensor([[1.,2.], [3.,4.]])
out1 = model(x1)
out2 = cmodel(x2)
torch.testing.assert_close(out1, out2)
