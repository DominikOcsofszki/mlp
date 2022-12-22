from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torchvision import datasets
import torch
import numpy as np


class MyDataSets:
    def __init__(self, tuble=(4, 9), batch_size_train=16, batch_size_test=10000):
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
    def __init__(self, batch_size_train=32, batch_size_test=10000):
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
        # _train_idx_0 = np.asarray(_trainset_full.targets == 0).nonzero()

        _train_idx = np.hstack(_train_idx_4 + _train_idx_9)
        # _train_idx = np.hstack(_train_idx_4 + _train_idx_9 + _train_idx_0)
        self.train_loader_subset_size = _size_train = len(_train_idx)
        print(f'{self.train_loader_subset_size = }')
        # print(f'{_train_idx = }')
        # print(f'{_size_train = }')
        _train_subset = torch.utils.data.Subset(_trainset_full, _train_idx)
        self.train_loader_subset = torch.utils.data.DataLoader(_train_subset, shuffle=True, batch_size=batch_size_train)

        # _test_idx = np.where(_testset_full.targets == (4 | 9))[0]

        _test_idx_4 = np.asarray(_testset_full.targets == 4).nonzero()
        _test_idx_9 = np.asarray(_testset_full.targets == 9).nonzero()
        # _test_idx_0 = np.asarray(_testset_full.targets == 0).nonzero()

        # _test_idx = np.hstack(_test_idx_4 + _test_idx_9 + _test_idx_0)
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

    # def for_plotting_dataloader_test_subset(self):
    #     return next(iter(self.test_loader_subset))
    def dataloader_train_subset(self):
        return self.train_loader_subset

    def dataloader_train_subset_one_batch(self):
        return next(iter(self.train_loader_subset))

    def dataloader_test_subset(self):
        return self.test_loader_subset

    def dataloader_test_subset_one_batch(self):
        return next(iter(self.test_loader_subset))

class MyDataSets_Subset_4_9:
    def __init__(self, batch_size_train=32, batch_size_test=10000):
        print('MyDataSets.MyDataSets_Subset_4_9.__init__')
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        # self.indices_batch_size_test_all = np.array([x for x in range(batch_size_test)])
        # print(f'{self.indices_batch_size_test_all}')

        # _dataset_train_full = datasets.MNIST(root='data/dataset', train=True, transform=transforms.ToTensor(),
        #                                          download=True)
        # _dataset_test_full = datasets.MNIST(root='data/testset', train=False, transform=transforms.ToTensor(),
        #                                         download=True)
        # self.dataloader_train_full = torch.utils.data.DataLoader(self.dataset_train_full, shuffle=True,
        #                                                          batch_size=self.batch_size_train)
        # self.dataloader_test_full = torch.utils.data.DataLoader(self.dataset_train_full, shuffle=True,
        #                                                         batch_size=self.batch_size_test)
        # self.test_subset_full = torch.utils.data.Subset(self.dataset_test_full, self.indices_batch_size_test_all)
        _trainset_full = datasets.MNIST(root='data/dataset', train=True, transform=transforms.ToTensor(),
                                        download=True)
        _testset_full = datasets.MNIST(root='data/testset', train=False, transform=transforms.ToTensor(),
                                       download=True)

        _train_idx_4 = np.asarray(_trainset_full.targets == 4).nonzero()
        _train_idx_9 = np.asarray(_trainset_full.targets == 9).nonzero()
        # Change class_nr: 4=>0, 9=>1
        _trainset_full.targets[_train_idx_4] = 0
        _trainset_full.targets[_train_idx_9] = 1

        _train_idx = np.hstack(_train_idx_4 + _train_idx_9)
        self.train_loader_subset_changed_labels_size = _size_train = len(_train_idx)
        print(f'{self.train_loader_subset_changed_labels_size = }')
        _train_subset_changed_labels_to_0_1 = torch.utils.data.Subset(_trainset_full, _train_idx)
        self.train_loader_subset_changed_labels = torch.utils.data.DataLoader(_train_subset_changed_labels_to_0_1,
                                                                              shuffle=True,
                                                                              batch_size=batch_size_train)

        # TEST
        _test_idx_4 = np.asarray(_testset_full.targets == 4).nonzero()
        _test_idx_9 = np.asarray(_testset_full.targets == 9).nonzero()

        # Change class_nr: 4=>0, 9=>1
        _testset_full.targets[_test_idx_4] = 0
        _testset_full.targets[_test_idx_9] = 1

        _test_idx = np.hstack(_test_idx_4 + _test_idx_9)

        self.test_loader_subset_changed_labels_size = len(_test_idx)
        _test_subset_changed_labels_to_0_1 = torch.utils.data.Subset(_testset_full, _test_idx)
        self.test_loader_subset_changed_labels = torch.utils.data.DataLoader(_test_subset_changed_labels_to_0_1,
                                                                             shuffle=True,
                                                                             batch_size=self.test_loader_subset_changed_labels_size)

        print(f'{self.train_loader_subset_changed_labels_size = }')
        print(f'{self.train_loader_subset_changed_labels = }')

        print(f'{self.test_loader_subset_changed_labels_size = }')
        print(f'{self.test_loader_subset_changed_labels = }')

    def for_plotting_dataloader_test_full(self):
        return next(iter(self.test_loader_subset_changed_labels))

    def dataloader_train_subset(self):
        return self.train_loader_subset_changed_labels

    def dataloader_train_subset_one_batch(self):
        return next(iter(self.dataloader_train_subset()))

    def dataloader_test_subset(self):
        return self.test_loader_subset_changed_labels

    def dataloader_test_subset_one_batch(self):
        return next(iter(self.dataloader_test_subset()))
