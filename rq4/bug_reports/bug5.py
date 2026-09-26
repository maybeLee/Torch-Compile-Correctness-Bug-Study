"""
Bug report for identifier: angle complex -0.0
Buggy Version: torch 2.12.0
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x[0] = complex(-0.0, 0.0)  # this line should be put inside forward to trigger this error
        out = torch.angle(x)
        return out

x1 = torch.tensor([1 + 1j], dtype=torch.complex64)
x2 = torch.tensor([1 + 1j], dtype=torch.complex64)
model = Model()
cmodel = torch.compile(model)
out1 = model(x1)
print(out1)  # tensor([3.1416])
out2 = cmodel(x2)
print(out2)  # tensor([0.])

torch.testing.assert_close(out1, out2)