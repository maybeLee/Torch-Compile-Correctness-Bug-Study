"""
Bug report for identifier: interpolate - align_corners=False
Buggy Version: torch 2.12.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 1.0/300.0

    def forward(self, x):
        y = F.interpolate(x, scale_factor=self.scale, mode='linear', align_corners=False)
        padded = F.pad(y, pad=(1, 0))
        out = torch.gt(padded, y)
        return out


torch.manual_seed(0)
x1 = torch.randn(1, 8, 396)
torch.manual_seed(0)
x2 = torch.randn(1, 8, 396)

model = Model()
cmodel = torch.compile(Model(), dynamic=True)
out1 = model(x1)
out2 = cmodel(x2)
torch.testing.assert_close(out1, out2)
