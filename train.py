import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

dataset = datasets.MNIST(root='data/dataset',train=True,transform=transforms.ToTensor(),download=True)
testset = datasets.MNIST(root='data/testset',transform=transforms.ToTensor(),download=True)
