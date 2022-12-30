import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class latent_space_old(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        # self.fc1 = nn.Linear(28 * 28, 20)
        # self.fc2 = nn.Linear(20, 2)
        self.fc = nn.Linear(2,2)
        # self.final = nn.BCEWithLogitsLoss()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        x = self.final(x)
        return x

class latent_space(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        # self.fc1 = nn.Linear(28 * 28, 20)
        # self.fc2 = nn.Linear(20, 2)
        self.fc = nn.Linear(2,1)
        # self.fc20 = nn.Linear(10,1)
        # self.final = nn.BCEWithLogitsLoss()
        self.activation = nn.Sigmoid()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = self.flatten(x)
        x = self.fc(x)
        # x = self.relu(self.fc20(x))
        x = self.relu(x)
        x = self.activation(x)
        return x