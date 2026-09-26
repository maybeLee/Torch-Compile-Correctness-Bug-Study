"""
Bug report for identifier: inf div
Buggy Version: torch 2.12.0
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a):
        b = torch.full_like(a, 0.25)
        return torch.div(a, b, rounding_mode='floor')


x = torch.tensor([float('inf')])

model = Model()
cmodel = torch.compile(model)

out1 = model(x)
print(out1)
out2 = cmodel(x)
print(out2)
torch.testing.assert_close(out1, out2, equal_nan=True)
