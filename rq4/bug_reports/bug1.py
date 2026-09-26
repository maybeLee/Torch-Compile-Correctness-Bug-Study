"""
Bug report for identifier: repeat -> as_strided
Buggy Version: torch 2.12.0
"""

import torch
def fn(x):
    full = x.repeat((3, 2))
    return torch.as_strided(full[:, ::2], (3, 2), full.stride())
torch.manual_seed(0)
x = torch.randn(1, 1)
out1 = fn(x)
# print(out1)
out2 = torch.compile(fn)(x)
# print(out2)
torch.testing.assert_close(out1, out2, equal_nan=True)

