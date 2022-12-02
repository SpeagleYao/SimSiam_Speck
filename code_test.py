import numpy as np
import torch
from crypto.speck import speck

sp = speck()
X, Y = sp.generate_train_data(10**7, 7)
print(X[Y==0].shape)
print(X[Y==1].shape)