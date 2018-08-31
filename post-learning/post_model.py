import torch.nn as nn
import torch
import torch.nn.functional as F


class PostNet(nn.Module):
    def __init__(self):
        super(PostNet, self).__init__()
        # first fc layer, separate
        self.fc1_1 = nn.Linear(128, 128)
        # self.fc1_2 = nn.Linear(128, 128)
        # then connected
        # self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(256, 128)
        # self.fc4 = nn.Linear(128, 1)
        # self.fc1_1 = nn.Linear(128, 32)
        # self.fc1_2 = nn.Linear(128, 32)
        # then connected
        # self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(256, 1)
        #
        # # self.fc3 = nn.Linear(256, 256)
        # self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 1)


    def forward(self, x, y):
        # drop = torch.nn.Dropout(0.2)

        z = x - y
        z = self.fc1_1(z)
        z = self.fc5(z)

        # x = F.relu(self.fc1_1(x))
        # # x = drop(x)
        # y = F.relu(self.fc1_2(y))
        # y = drop(y)
        # z = torch.cat((x, y), dim=1)
        #
        # z = torch.nn.functional.tanh(self.fc2(z))
        # # z = drop(z)
        # z = torch.nn.functional.tanh(self.fc3(z))
        # z = torch.nn.functional.tanh(self.fc4(z))
        # z = self.fc2(z)
        # z = self.fc3(z)

        # return torch.nn.functional.tanh(z)
        return z