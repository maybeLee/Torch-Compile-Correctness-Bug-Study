"""
Bug report for identifier: unsqueeze - permute - vector_norm
Buggy Version: torch 2.12.0
Already fixed
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.unfold(0, 2, 1)
        x = torch.unsqueeze(x, 1)
        x = torch.permute(x, [0, 2, 3, -3])
        last_dim = x.size(-1)
        out = torch.linalg.vector_norm(x, ord=2, dim=[-1])
        out = out / (last_dim if last_dim != 0 else 1)
        return (out,)



torch.manual_seed(0)
x = torch.randn(4, 4, dtype=torch.float32)
model = Model()
cmodel = torch.compile(model)
out1 = model(x)
out2 = cmodel(x)
torch.testing.assert_close(out1, out2)
