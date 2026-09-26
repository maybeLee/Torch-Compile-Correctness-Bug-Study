"""
Bug report for identifier: w.data.fill_ + register_buffer+=1 + pow
Buggy Version: torch 2.12.0
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("_ccc", torch.zeros((), dtype=torch.int64))

    def forward(self, x):
        self.w.data.fill_(0.0)
        self._ccc += 1
        return torch.pow(x + 0, 1)

model = Model()
cmodel = torch.compile(Model())
x = torch.tensor([1.,2.,3.,4.])
out1 = model(x)
out2 = cmodel(x)
print(out1)  # tensor([1., 2., 3., 4.])
print(out2)  # tensor(1)
torch.testing.assert_close(out1, out2)
