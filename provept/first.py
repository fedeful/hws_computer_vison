from __future__ import print_function
import torch

x = torch.Tensor(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

y = torch.rand(5, 3)
print(y)

print(x+y)

print(x.t_())

print(x[:, 1])


l = torch.randn(4, 4)
y = l.view(16)
z = l.view(-1,8)
print(l,y,z)