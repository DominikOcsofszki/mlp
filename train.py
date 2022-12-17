import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch.utils.data

# Variables
BATCH_SIZE = 64

# Downloading the dataset
trainset = datasets.MNIST(root='data/dataset', train=True, transform=transforms.ToTensor(), download=True)
testset = datasets.MNIST(root='data/testset', transform=transforms.ToTensor(), download=True)

# Filter for only two classes #TODO Not sure yet if it is needed
#TODO Do we need to normalize the data, or is it already done?
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testsetloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

# Trainloader
