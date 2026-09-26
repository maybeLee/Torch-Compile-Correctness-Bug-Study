"""
Bug report for identifier: negative * 0.0 + 0.0
Buggy Version: torch 2.12.0
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.0 + 0.0

model = Model()
cmodel = torch.compile(model)

x1 = torch.tensor([-1.0, -2.0, 3.0])
x2 = torch.tensor([-1.0, -2.0, 3.0])
out1 = model(x1)
out2 = cmodel(x2)
torch.testing.assert_close(out1, out2)
