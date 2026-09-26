"""
Bug report for identifier: nan/inf * 0
Buggy Version: torch 2.12.0
"""
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0

torch.manual_seed(0)
x = torch.randn(8, 8) * torch.inf

model = Model()
cmodel = torch.compile(model)
out1 = model(x)
out2 = cmodel(x)
torch.testing.assert_close(out1, out2, equal_nan=True)

