"""
Bug report for identifier: logaddexp complex inf
Buggy Version: torch 2.12.0
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return torch.logaddexp(x, y)

x = torch.tensor([[complex(float('inf'), 0.0), complex(1.0, -2.0)],
                    [complex(-3.0, 4.0), complex(0.0, float('-inf'))]], dtype=torch.complex64)
y = torch.tensor([[complex(0.0, 0.0), complex(float('-inf'), 1.0)],
                    [complex(2.0, -1.0), complex(float('inf'), float('inf'))]], dtype=torch.complex64)
model = Model()
cmodel = torch.compile(model)

out1 = model(x, y)
print(out1)
out2 = cmodel(x, y)
print(out2)
torch.testing.assert_close(out1, out2, equal_nan=True)