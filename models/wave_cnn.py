import torch
import torch.nn as nn
import torch.nn.functional as F

class Wave_CNN(nn.Module):
    def __init__(self, wt):
        super(Wave_CNN, self).__init__()
        self.fc1 = nn.Linear(4909, 50)
        self.fc2 = nn.Linear(50, 10)
        self.wt = wt.eval()
        self.wt.J = 3
        # freeze layers
        for param in wt.parameters():
            param.requires_grad = False

    def forward(self, x):
        batch_size = x.shape[0]
        x_t = self.wt(x)

        x_t0 = x_t[0]
        x_t1 = x_t[1]
        x_t2 = x_t[2]
        x_t3 = x_t[3]

        self.wt.J = 2
        x_t1 = self.wt(F.relu(x_t1.squeeze()))
        self.wt.J = 1
        x_t2 = self.wt(F.relu(x_t2.squeeze()))

        x = []
        x.append(x_t0.reshape(batch_size, -1))
        for j in range(len(x_t1)):
            x.append(x_t1[j].reshape(batch_size, -1))
        for j in range(len(x_t2)):
            x.append(x_t2[j].reshape(batch_size, -1))
        x.append(x_t3.reshape(batch_size, -1))
        x = torch.cat(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        self.wt.J = 3
        return x