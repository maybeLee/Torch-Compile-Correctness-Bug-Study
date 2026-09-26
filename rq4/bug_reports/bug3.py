"""
Bug report for identifier: -0.0 -> reciprocal/div
Buggy Version: torch 2.12.0
"""


import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x):
        zero = x * 0
        return torch.reciprocal(zero)

torch.manual_seed(0)
x = torch.randn(8)

model = Model()
cmodel = torch.compile(model)
out1 = model(x)
out2 = cmodel(x)
torch.testing.assert_close(out1, out2)
