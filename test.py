import numpy as np
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models import *
from tqdm import tqdm
from crypto.speck import speck
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

# Test settings
parser = argparse.ArgumentParser(description='PyTorch Model Train')
parser.add_argument('--test-batch-size', type=int, default=5000,
                help='input batch size for testing (default: 5000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                help='disables CUDA training')
parser.add_argument('--nr', type=int, default=7,
                help='round of encryptions (default: 7)')
args = parser.parse_args()

sp = speck()
X_test, Y_test = sp.generate_train_data(10**6, args.nr)
X_test = torch.as_tensor(X_test).to(torch.float32)
Y_test = torch.as_tensor(Y_test).to(torch.float32)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=args.test_batch_size, num_workers=4, pin_memory=True)

criterion = nn.CosineSimilarity(dim=1).cuda()

model = SS().cuda()
model.load_state_dict(torch.load('./checkpoints/SimSiam_'+str(args.nr)+'r.pth'))

model.eval()
test_loss_t = []
test_loss_f = []
with torch.no_grad():
    for (data, target) in tqdm(test_loader):
        data = data.cuda();target = target.cuda()
        p1, p2, z1, z2 = model(data)
        p1t = p1[target==1];p2t = p2[target==1];z1t = z1[target==1];z2t = z2[target==1];
        p1f = p1[target==0];p2f = p2[target==0];z1f = z1[target==0];z2f = z2[target==0];
        loss_t = -(criterion(p1t, z2t) + criterion(p2t, z1t)) * 0.5
        loss_f = -(criterion(p1f, z2f) + criterion(p2f, z1f)) * 0.5
        # print(loss_t)
        # break
        test_loss_t.extend(loss_t.cpu().detach())
        test_loss_f.extend(loss_f.cpu().detach())
    tqdm.write('Test Set: True Loss: {:.6f}\tFalse Loss: {:.6f}'.format(np.mean(test_loss_t), np.mean(test_loss_f)))

TPR = []
TNR = []
ACC = []
x = []
DIV = 1000
test_loss_t = np.array(test_loss_t)
test_loss_f = np.array(test_loss_f)
for i in range(DIV):
    x.append(-1 + i / DIV)
    TPR.append((test_loss_t<x[i]).sum()/test_loss_t.shape)
    TNR.append((test_loss_f>x[i]).sum()/len(test_loss_f))
    ACC.append(((test_loss_t<x[i]).sum()+(test_loss_f>x[i]).sum())/(len(test_loss_t)+len(test_loss_f)))
plt.plot(x, TPR, label = 'TPR')
plt.plot(x, TNR, label = 'TNR')
plt.plot(x, ACC, label = 'ACC')
plt.legend()
plt.grid()
plt.savefig('./test_result/acc_'+str(args.nr)+'r.png')

# plt.plot(test_loss_t, 'r.', markersize=1, label = 'True')
# plt.plot(test_loss_f, 'b.', markersize=1, label = 'False')
# plt.legend()
# plt.plot()
# plt.savefig('./test_result/dist_'+str(args.nr)+'r.png')

