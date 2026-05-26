import torch
v1 = torch.tensor([float('nan'), 1.0])
v2 = torch.tensor([float('nan'), 1.0])
print(torch.allclose(v1, v2))
print(torch.allclose(v1, v2, equal_nan=True))
