"""
Bug report for identifier: expand -> index_put_
Buggy Version: torch 2.12.0
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, a):
        base = a[:1]
        x = base.expand(3)
        mask = torch.tensor([True, False, True])
        idx = [torch.tensor([3, 1, 2])]
        x.index_put_((mask,), torch.tensor([10.0, 20.0]))
        return base, x

torch.manual_seed(0)
x1 = torch.randn(8)
torch.manual_seed(0)
x2 = torch.randn(8)

model = Model()
cmodel = torch.compile(model)

out1 = model(x1)
print(out1)
out2 = cmodel(x2)
print(out2)
torch.testing.assert_close(out1, out2)
