import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(4, 10, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(10, 28, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        # self.conv1 = nn.Conv2d(in_channels=28, kernel_size=(3, 3), stride=1, padding=1, out_channels=28)
        # self.conv2 = nn.Conv2d(in_channels=28 * 28, kernel_size=(3, 3), stride=1, padding=1, out_channels=28 * 28)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(10*28*28,10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        # print(x.shape)
        x = self.relu(self.conv2(x))
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        return x
