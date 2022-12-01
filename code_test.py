import numpy as np
import torch

a = torch.as_tensor(np.arange(640))
a = a.view(10, 4, 16)
print(a[:, 0:2, :])
print(a[:, 2:4, :].shape)