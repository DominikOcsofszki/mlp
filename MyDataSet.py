
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torchvision import datasets
import torch
import numpy as np

class MyDataSets:
    def __init__(self, tuble=(4, 9), batch_size_train=16,batch_size_test=10000):
        print('MyDataSets.MyDataSets.__init__')
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.indices_batch_size_test_all = np.array([x for x in range(batch_size_test)])
        print(f'{self.indices_batch_size_test_all}')
        self.dataset_train_full = datasets.MNIST(root='data/dataset', train=True, transform=transforms.ToTensor(),
                                                 download=True)
        self.dataset_test_full = datasets.MNIST(root='data/testset', train=False, transform=transforms.ToTensor(),
                                                download=True)
        self.dataloader_train_full = torch.utils.data.DataLoader(self.dataset_train_full, shuffle=True,
                                                                 batch_size=self.batch_size_train)
        self.dataloader_test_full = torch.utils.data.DataLoader(self.dataset_train_full, shuffle=True,
                                                                batch_size=self.batch_size_test)
        self.test_subset_full = torch.utils.data.Subset(self.dataset_test_full, self.indices_batch_size_test_all)
        _trainset_full = datasets.MNIST(root='data/dataset', train=True, transform=transforms.ToTensor(), download=True)
        _testset_full = datasets.MNIST(root='data/testset', train=False, transform=transforms.ToTensor(), download=True)

        _train_idx_4 = np.asarray(_trainset_full.targets == 4).nonzero()
        _train_idx_9 = np.asarray(_trainset_full.targets == 9).nonzero()

        self.train_loader_subset_size = _train_idx = np.hstack(_train_idx_4 + _train_idx_9)
        _size_train = len(_train_idx)
        # print(f'{_train_idx = }')
        # print(f'{_size_train = }')
        _train_subset = torch.utils.data.Subset(_trainset_full, _train_idx)
        self.train_loader_subset = torch.utils.data.DataLoader(_train_subset, shuffle=True, batch_size=_size_train)

        # _test_idx = np.where(_testset_full.targets == (4 | 9))[0]

        _test_idx_4 = np.asarray(_testset_full.targets == 4).nonzero()
        _test_idx_9 = np.asarray(_testset_full.targets == 9).nonzero()

        _test_idx = np.hstack(_test_idx_4 + _test_idx_9)

        # train_idx = np.where(testset.targets == tuble)[0]
        self.test_loader_subset_size = _size_test = len(_test_idx)
        _test_subset = torch.utils.data.Subset(_testset_full, _test_idx)
        self.test_loader_subset = torch.utils.data.DataLoader(_test_subset, shuffle=True, batch_size=_size_test)
        # self.test_loader_subset_size = (self.test_loader_subset).l

        print(f'{self.train_loader_subset_size = }')
        print(f'{self.train_loader_subset = }')
        print(f'{self.test_loader_subset_size = }')

    def for_plotting_dataloader_test_full(self):
        return next(iter(self.dataloader_test_full))

class MyDataSets_Subset:
    def __init__(self, tuble=(4, 9), batch_size_train=16,batch_size_test=10000):
        print('MyDataSets.MyDataSets_Subset.__init__')
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.indices_batch_size_test_all = np.array([x for x in range(batch_size_test)])
        print(f'{self.indices_batch_size_test_all}')
        self.dataset_train_full = datasets.MNIST(root='data/dataset', train=True, transform=transforms.ToTensor(),
                                                 download=True)
        self.dataset_test_full = datasets.MNIST(root='data/testset', train=False, transform=transforms.ToTensor(),
                                                download=True)
        self.dataloader_train_full = torch.utils.data.DataLoader(self.dataset_train_full, shuffle=True,
                                                                 batch_size=self.batch_size_train)
        self.dataloader_test_full = torch.utils.data.DataLoader(self.dataset_train_full, shuffle=True,
                                                                batch_size=self.batch_size_test)
        self.test_subset_full = torch.utils.data.Subset(self.dataset_test_full, self.indices_batch_size_test_all)
        _trainset_full = datasets.MNIST(root='data/dataset', train=True, transform=transforms.ToTensor(), download=True)
        _testset_full = datasets.MNIST(root='data/testset', train=False, transform=transforms.ToTensor(), download=True)

        _train_idx_4 = np.asarray(_trainset_full.targets == 4).nonzero()
        _train_idx_9 = np.asarray(_trainset_full.targets == 9).nonzero()

        self.train_loader_subset_size = _train_idx = np.hstack(_train_idx_4 + _train_idx_9)
        _size_train = len(_train_idx)
        # print(f'{_train_idx = }')
        # print(f'{_size_train = }')
        _train_subset = torch.utils.data.Subset(_trainset_full, _train_idx)
        self.train_loader_subset = torch.utils.data.DataLoader(_train_subset, shuffle=True, batch_size=_size_train)

        # _test_idx = np.where(_testset_full.targets == (4 | 9))[0]

        _test_idx_4 = np.asarray(_testset_full.targets == 4).nonzero()
        _test_idx_9 = np.asarray(_testset_full.targets == 9).nonzero()

        _test_idx = np.hstack(_test_idx_4 + _test_idx_9)

        # train_idx = np.where(testset.targets == tuble)[0]
        self.test_loader_subset_size = _size_test = len(_test_idx)
        _test_subset = torch.utils.data.Subset(_testset_full, _test_idx)
        self.test_loader_subset = torch.utils.data.DataLoader(_test_subset, shuffle=True, batch_size=_size_test)
        # self.test_loader_subset_size = (self.test_loader_subset).l

        print(f'{self.train_loader_subset_size = }')
        print(f'{self.train_loader_subset = }')
        print(f'{self.test_loader_subset_size = }')

    def for_plotting_dataloader_test_full(self):
        return next(iter(self.dataloader_test_full))