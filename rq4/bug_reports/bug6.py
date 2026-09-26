"""
Bug report for identifier: expand + fill_
Buggy Version: torch 2.12.0
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, base):
        v = base.expand(3, -1)
        v.fill_(2.0)
        return base

x1 = torch.tensor([-1.0, -0.5, 0.0, 0.5])
x2 = torch.tensor([-1.0, -0.5, 0.0, 0.5])
model = Model()
cmodel = torch.compile(model)

out1 = model(x1)
print(out1)
out2 = cmodel(x2)
print(out2)
torch.testing.assert_close(out1, out2)

