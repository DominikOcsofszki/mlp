import model
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import datasets
import torch
from model import MyModel5
import numpy as np
import torch.nn as nn
from MyDataSet import MyDataSets


# (class_nr_0 = False,class_nr_1 = False,class_nr_2 = False,class_nr_3 = False,
#                class_nr_4 = False,class_nr_5 = False,class_nr_6 = False,class_nr_7 = False,
#                class_nr_8 = False,class_nr_9 = False, all_classes=False)

class MyDataSets_old:
    def __init__(self, all_classes=False, tuble=(4, 9)):
        trainset = datasets.MNIST(root='data/dataset', train=True, transform=transforms.ToTensor())
        testset = datasets.MNIST(root='data/testset', train=False, transform=transforms.ToTensor())
        if all_classes:
            self.trainset = trainset
            self.testset = testset
            self.trainset_size = trainset.__len__()
            self.testset_size = testset.__len__()
        else:
            indices_train = []
            indices_test = []
            for class_nr in tuble:
                indices_train.append((trainset.targets == class_nr).nonzero())
                indices_test.append((testset.targets == class_nr).nonzero())
            indices_train = torch.vstack(indices_train).reshape(-1)
            indices_test = torch.vstack(indices_test).reshape(-1)
            filtered_trainset = torch.utils.data.Subset(dataset=trainset, indices=indices_train)
            torch.hstack(filtered_trainset)
            filtered_testset = torch.utils.data.Subset(dataset=testset, indices=indices_test)
            ##Convert to Tensor
            Tensortrain, Tensortrain_label = [(x[0], x[1]) for x in filtered_trainset]
            #############
            self.trainset = filtered_trainset
            self.testset = filtered_testset

            self.trainset_size = indices_train.size()[0]
            self.testset_size = indices_test.size()[0]
        print(f'{self.trainset_size = }')
        print(f'{self.testset_size = }')


class MyDataSets_changed_in_new_class:
    def __init__(self, tuble=(4, 9), batch_size_train=16, batch_size_test=10000):
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.dataset_train_full = datasets.MNIST(root='data/dataset', train=True, transform=transforms.ToTensor(),
                                                 download=True)
        self.dataset_test_full = datasets.MNIST(root='data/testset', train=False, transform=transforms.ToTensor(),
                                                download=True)
        self.dataloader_train_full = torch.utils.data.DataLoader(self.dataset_train_full, shuffle=True,
                                                                 batch_size=self.batch_size_train)
        self.dataloader_test_full = torch.utils.data.DataLoader(self.dataset_train_full, shuffle=True,
                                                                batch_size=self.batch_size_test)

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


def import_model_name(model_x: nn.Module, activate_eval=True):
    model_name = model_x._get_name()
    print(f'{model_name = }')
    save_name_model = model_name + '_weights'
    PATH = '/Users/dominikocsofszki/PycharmProjects/mlp/data/weights/' + save_name_model
    model_x.load_state_dict(torch.load(PATH))
    if activate_eval: model_x.eval()
    print(f'{save_name_model} imported')
    return model_x


def return_images_labels(count_of_images=5, training_set=False):
    testset = datasets.MNIST(root='data/testset', transform=transforms.ToTensor(), download=True,
                             train=training_set)
    class_4_test = (testset.targets == 4)
    class_9_test = (testset.targets == 9)

    indices_test = (class_4_test | class_9_test).nonzero().reshape(-1)

    # filtered_trainset = torch.utils.data.Subset(dataset=trainset, indices=indices_train)
    filtered_testset = torch.utils.data.Subset(dataset=testset, indices=indices_test)

    # --------------------------

    # trainset = filtered_trainset
    testset = filtered_testset

    testsetloader = torch.utils.data.DataLoader(testset, batch_size=count_of_images, shuffle=True)  # TODO shuffle for
    testing_images, labels = next(iter(testsetloader))
    return testing_images, labels


def show_images_with_model(count_of_images=5, model=None, only_return_images_labels=False, training_set=False):
    if model is None:
        PATH_weight_classify = '/Users/dominikocsofszki/PycharmProjects/mlp/data/weights/weights_model_classifier'
        model_classify = MyModel5()
        model_classify.load_state_dict(torch.load(PATH_weight_classify))
        model_classify.eval()
        model = model_classify
    model.eval()
    testset = datasets.MNIST(root='data/testset', transform=transforms.ToTensor(), download=True,
                             train=training_set)
    testsetloader = torch.utils.data.DataLoader(testset, batch_size=count_of_images, shuffle=True)  # TODO shuffle for
    testing_images, labels = next(iter(testsetloader))
    if only_return_images_labels:
        return testing_images, labels
    if count_of_images == 1: num_of_tests = 2
    num_of_tests = testing_images.__len__()
    size_fig = 15
    num_plots = count_of_images
    if count_of_images == 1: num_plots = 2
    if count_of_images <= 8: size_fig = 150 / count_of_images
    if count_of_images <= 3: size_fig = 5
    if count_of_images > 8: size_fig = 200 / count_of_images
    if count_of_images > 15: size_fig = 300 / count_of_images
    fig, axs = plt.subplots(1, num_plots, figsize=(size_fig, size_fig))
    # fig2, axs2 = plt.subplots(1, num_plots, figsize=(size_fig, size_fig))
    # print(f'{num_of_tests = }')
    PRED_bool = True
    # print(xx)
    if PRED_bool:
        pred = model(testing_images)
        # print(pred.shape)
        # print(pred)
    for indx in range(num_of_tests):
        title = str(int(labels[indx])) + '\npred:'
        if PRED_bool:
            pred_acc = pred[indx]
            pred_nr = int(pred_acc.argmax())
            acc = pred_acc[pred_nr]
            title += str(pred_nr) + ' '
            # title += str(int(acc)) # TODO add later as accuracy

        axs[indx].set_yticklabels([])  # x-axis
        axs[indx].set_xticklabels([])  # y-axis
        axs[indx].imshow(testing_images[indx, 0, :, :])
        axs[indx].set_title(title)
        # axs[indx].set_title(int(labels[indx]))

    # for indx in range(num_of_tests):
    #     axs2[indx].set_yticklabels([])  # x-axis
    #     axs2[indx].set_xticklabels([])  # y-axis
    #     pred = model(testing_images[indx, 0, :, :])
    #     axs2[indx].imshow(pred)
    #     # axs2[indx].imshow(testing_images[indx, 0, :, :])
    #     axs2[indx].set_title(int(labels[indx]))
    return testing_images, labels


def print_modul_entries(model):
    for x in model.modules():
        print(x)


def print_sth_once_ret_new_count(sth_to_print, COUNT_PRINTS, add_text=''):
    if COUNT_PRINTS == 0:
        return 0
    else:
        # print(add_text)
        print(f'{add_text} : {sth_to_print:}')
        # COUNTER_TOP-=1
        return COUNT_PRINTS - 1


def counter_for_runs(reset_to_zero=False):
    x = 0
    if not reset_to_zero:
        with open('/Users/dominikocsofszki/PycharmProjects/mlp/data/weights/counter', 'w+') as f:
            x = f.readline()
            print(x)
            x = int(x)

            x += 1
            print(x)
            f.write(str(x))
    return x


def show_images_with_model_new(count_of_images=5, model=None):
    if model is None:
        PATH_weight_classify = '/Users/dominikocsofszki/PycharmProjects/mlp/data/weights/weights_model_classifier'
        # PATH_weight_classify = '/Users/dominikocsofszki/PycharmProjects/mlp/data/weights/MyModel5_weights'
        model_classify = MyModel5()
        model_classify.load_state_dict(torch.load(PATH_weight_classify))
        model_classify.eval()
        model = model_classify
    model.eval()
    testset = datasets.MNIST(root='data/testset', train=False, transform=transforms.ToTensor(), download=True)
    testsetloader = torch.utils.data.DataLoader(testset, batch_size=count_of_images, shuffle=True)  # TODO shuffle for
    testing_images, labels = next(iter(testsetloader))
    if count_of_images == 1: num_of_tests = 2
    num_of_tests = testing_images.__len__()
    size_fig = 15
    num_plots = count_of_images
    if count_of_images == 1: num_plots = 2
    if count_of_images <= 8: size_fig = 150 / count_of_images
    if count_of_images <= 3: size_fig = 5
    if count_of_images > 8: size_fig = 200 / count_of_images
    if count_of_images > 15: size_fig = 300 / count_of_images
    fig, axs = plt.subplots(1, num_plots, figsize=(size_fig, size_fig))
    fig, axs2 = plt.subplots(1, num_plots, figsize=(size_fig, size_fig))
    PRED_bool = True
    for axxx in axs:
        axxx.set_yticklabels([])  # x-axis
        axxx.set_xticklabels([])  # y-axis
    for axxx in axs2:
        axxx.set_yticklabels([])  # x-axis
        axxx.set_xticklabels([])  # y-axis
    # print(xx)
    if PRED_bool:
        pred = model(testing_images)
    for indx in range(num_of_tests):
        title = str(int(labels[indx])) + '\npred:'
        if PRED_bool:
            pred_acc = pred[indx]
            pred_nr = int(pred_acc.argmax())
            acc = pred_acc[pred_nr]
            title += str(pred_nr) + ' '
            # title += str(int(acc)) # TODO add later as accuracy

        axs[indx].imshow(testing_images[indx, 0, :, :])
        axs[indx].set_title(title)

    pred_alt = model.forward(testing_images)
    for indx in range(num_of_tests):
        nump_pred = pred_alt[indx].view(28, 28).detach().numpy()
        print(nump_pred.shape)
        axs2[indx].imshow(nump_pred)
        axs2[indx].set_title(int(labels[indx]))
    return testing_images, labels


def show_scatter_lattent(examples, model_loaded, cmap='Dark2'):
    # model_loaded = model.VaeMe_200_hidden()
    mymodel = import_model_name(model_x=model_loaded, activate_eval=True)
    images, labels = show_images_with_model(examples, model=model_loaded, only_return_images_labels=True)
    z = mymodel.forward_return_z(images)
    # print(z[0])
    z_detached = z.detach().numpy()
    labels_detached = labels.detach().numpy()
    z_detached = np.column_stack((z_detached, labels_detached))
    # print(f'{labels_detached.shape = }')
    plt.figure(figsize=(10, 10))
    plt.scatter(z_detached[:, 0], z_detached[:, 1], c=labels, cmap=cmap)  # ToDo looks nice!
    cbar = plt.colorbar()
    cbar.set_label('labels')
    plt.show()


def show_scatter_lattent_3D(examples, model_loaded, cmap='Dark2'):
    # model_loaded = model.VaeMe_200_hidden()
    mymodel = import_model_name(model_x=model_loaded, activate_eval=True)
    images, labels = show_images_with_model(examples, model=model_loaded, only_return_images_labels=True)
    z = mymodel.forward_return_z(images)
    z_detached = z.detach().numpy()
    plt.figure(figsize=(10, 10))
    plt.scatter(z_detached[:, 0], z_detached[:, 1], z_detached[:, 2], c=labels, cmap=cmap)  # ToDo looks nice!
    plt.colorbar()
    plt.show()


def show_scatter_lat(examples, model_loaded, lat: int = 2, cmap='Dark2', train=False, use_training_set=False):
    if not train: model_loaded = import_model_name(model_x=model_loaded, activate_eval=True)
    images, labels = show_images_with_model(examples, model=model_loaded, only_return_images_labels=True,
                                            training_set=use_training_set)
    z = model_loaded.forward_return_z(images)
    # z = model_loaded.encode(images)
    # print(f'{z.shape = }')
    # print(z)
    z_detached = z.detach().numpy()
    plt.figure(figsize=(10, 10))
    if lat == 2:
        plt.scatter(z_detached[:, 0], z_detached[:, 1], c=labels, cmap=cmap)  # ToDo looks nice!
    else:
        if lat == 3:
            plt.scatter(z_detached[:, 0], z_detached[:, 1], z_detached[:, 2], c=labels,
                        cmap=cmap)  # ToDo looks nice!
        else:
            print('lat == 2|3')

    plt.colorbar()
    plt.show()


def show_scatter_lat_mu_sigma(examples, model_loaded, lat: int = 2, cmap='Dark2', in_train_class=False,
                              use_training_set=False):
    if not in_train_class: model_loaded = import_model_name(model_x=model_loaded, activate_eval=True)
    images, labels = show_images_with_model(examples, model=model_loaded, only_return_images_labels=True,
                                            training_set=use_training_set)
    mu, sigma = model_loaded.encode(images)
    # z = model_loaded.calc_z(mu, sigma)
    # print(mu,sigma)
    z = mu + sigma
    # print(f'{mu[0]=}')
    # print(f'{sigma[0]=}')

    # print(f'{z[0]=}')
    z_det = z.detach().numpy()
    print(f'{z.mean() = }')
    plt.figure(figsize=(10, 10))
    if lat == 2:
        plt.scatter(z_det[:, 0], z_det[:, 1], c=labels, cmap=cmap, alpha=0.9)  # ToDo looks nice!
    if lat == 3:
        plt.scatter(z_det[:, 0], z_det[:, 1], z_det[:, 2], c=labels, cmap=cmap, alpha=0.5)  # ToDo looks nice!
    plt.colorbar()
    plt.show()


def show_scatter(examples, model_loaded, lat: int = 2, cmap='Dark2', in_train_class=False, use_training_set=False,
                 current_epoch='epoch_information'):
    if not in_train_class: model_loaded = import_model_name(model_x=model_loaded, activate_eval=True)
    images, labels = show_images_with_model(examples, model=model_loaded, only_return_images_labels=True,
                                            training_set=use_training_set)
    mu, sigma = model_loaded.encode(images)
    # z = mu + sigma
    z = mu
    z_det = z.detach().numpy()
    plt.figure(figsize=(10, 10))
    plt.scatter(z_det[:, 0], z_det[:, 1], c=labels, cmap=cmap, alpha=0.9)  # ToDo looks nice!
    plt.colorbar()
    plt.title(f'{current_epoch = }')
    plt.show()


def show_scatter_binary_train(model_loaded, mydataset: MyDataSets, current_epoch='epoch_information'):
    size = mydataset.testset
    testsetloader = torch.utils.data.DataLoader(mydataset.testset_size, batch_size=size)  # TODO shuffle for
    testing_images, labels = next(iter(testsetloader))

    # images, labels = return_images_labels(examples, training_set=use_training_set)
    mu, sigma = model_loaded.encode(images)
    z = mu
    z_det = z.detach().numpy()
    plt.figure(figsize=(10, 10))
    plt.scatter(z_det[:, 0], z_det[:, 1], c=labels, cmap='', alpha=0.9)  # ToDo looks nice!
    plt.colorbar()
    plt.title(f'{current_epoch = }')
    plt.show()


def show_scatter_binary_dataset(model: nn.Module, mydataset: MyDataSets, current_epoch='epoch_information'):
    model.eval()

    testing_images, labels = mydataset.for_plotting_dataloader_test_full()
    mu, sigma = model.encode(testing_images)
    # print(f'scatter_plot: {mu}')
    z = mu
    print(f'{z.mean() = }')
    z_det = z.detach().numpy()
    model.train()
    plt.figure(figsize=(10, 10))
    # plt.scatter(z_det[:, 0], z_det[:, 1], c=labels, cmap='Dark2_r', alpha=0.9)
    plt.scatter(z_det[:, 0], z_det[:, 1], c=labels, cmap='tab10', alpha=0.9)
    plt.colorbar()
    plt.title(f'{current_epoch = }')
    plt.show()


def latent_to_plt_img(modelpick, rand_lat_size=2):
    # rand_lat = torch.rand(2)
    rand_lat = torch.rand(rand_lat_size)
    z = modelpick.decode(rand_lat)
    print(rand_lat.shape)
    print(f'{rand_lat = }')

    z_reshaped = z.view(28, 28)
    print(z.shape)
    print(z_reshaped.shape)

    znew = z_reshaped.detach()
    plt.imshow(znew)


def latent_rand_to_img(modelpick, tensor_rand):
    rand_lat = tensor_rand
    z = modelpick.decode(rand_lat)
    z_reshaped = z.view(28, 28)
    znew = z_reshaped.detach()
    plt.imshow(znew)


def print_imported_functions():
    print('imported Functions:')
    print('def import_model_name(model_x, activate_eval=True):')
    print('def show_images_with_model(count_of_images=5, model=None):')
    print('def print_modul_entries(model):')
    print('print_sth_once_ret_new_count(sth_to_print, COUNT_PRINTS)')
    print('counter_for_runs(reset_to_zero=False):')


if __name__ == '__main__':
    #     print('in helper-main:')
    #     count = 99
    #     # print(count)
    #     count = counter_for_runs()
    #     print(count)
    model_loaded = model.VaeMe_200_hidden()
    # model_loaded = model.VaeMe()
    mymodel = import_model_name(model_x=model_loaded, activate_eval=True)
    # images, labels = helper.show_images_with_model(20,model = model_loaded)
    images, labels = show_images_with_model_new(20, model=model_loaded)
    vec = mymodel(images)
    z = mymodel.forward_return_z(images)
    indx = 0
    for item in z:
        print(int(labels[indx]), item)
        indx += 1
