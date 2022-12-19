import model
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import datasets
import torch
from model import MyModel5
import numpy as np


def import_model_name(model_x, activate_eval=True):
    model_name = model_x._get_name()
    print(f'{model_name = }')
    save_name_model = model_name + '_weights'
    PATH = '/Users/dominikocsofszki/PycharmProjects/mlp/data/weights/' + save_name_model
    model_x.load_state_dict(torch.load(PATH))
    if activate_eval: model_x.eval()
    print(f'{save_name_model} imported')
    return model_x


def show_images_with_model(count_of_images=5, model=None, only_return_images_labels=False):
    if model is None:
        PATH_weight_classify = '/Users/dominikocsofszki/PycharmProjects/mlp/data/weights/weights_model_classifier'
        # PATH_weight_classify = '/Users/dominikocsofszki/PycharmProjects/mlp/data/weights/MyModel5_weights'
        model_classify = MyModel5()
        model_classify.load_state_dict(torch.load(PATH_weight_classify))
        model_classify.eval()
        model = model_classify
    model.eval()
    testset = datasets.MNIST(root='data/testset', transform=transforms.ToTensor(), download=True,train=False)   ###TODO MISSED THE TRAIN!!!
    print(f"{testset.__len__() = }")
    testsetloader = torch.utils.data.DataLoader(testset, batch_size=count_of_images, shuffle=True)  # TODO shuffle for
    testing_images, labels = next(iter(testsetloader))
    if only_return_images_labels: return testing_images, labels
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
    testset = datasets.MNIST(root='data/testset', transform=transforms.ToTensor(), download=True)
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

    z_detached = z.detach().numpy()
    labels_detached = labels.detach().numpy()
    z_detached = np.column_stack((z_detached, labels_detached))
    print(f'{labels_detached.shape = }')
    plt.figure(figsize=(10, 10))
    # plt.scatter(z_detached[:,0],z_detached[:,1],c=labels, cmap='Dark2') #ToDo looks nice!
    plt.scatter(z_detached[:,0],z_detached[:,1],c=labels, cmap=cmap) #ToDo looks nice!
    # plt.scatter(z_detached[:, 0], z_detached[:, 1], c=labels)
    plt.colorbar()
    plt.show()


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
