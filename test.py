import torch


c = torch.randint(0,10, (10,1)).to(torch.float32)
print(c)
c = c[None:4,:]
print(c)
