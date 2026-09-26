"""
Bug report for identifier: expand - index_put ([0,0]) - index_get ([0])
Buggy Version: torch 2.12.0
"""


import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b = x.expand(2, *x.shape)
        b[0, 0] = 1.0
        idx = (b[0].to(torch.int64))
        return idx


torch.manual_seed(0)
x1 = torch.tensor([[ -1.5410, -0.2934],
        [-2.1788,  0.5684]])
torch.manual_seed(0)
x2 = torch.tensor([[ -1.5410, -0.2934],
        [-2.1788,  0.5684]])
model = Model()
cmodel = torch.compile(Model())
out1 = model(x1)
print(out1)
'''
tensor([[ 1,  1],
        [-2,  0]])
'''
out2 = cmodel(x2)
'''
tensor([[ 0,  1],
        [-2,  0]])
'''
print(out2)
torch.testing.assert_close(out1, out2)
