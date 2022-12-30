import os

import mlflow
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

import MyDataSet
import helper
import model
import model_probit
from MyDataSet import MyDataSets_Subset, MyDataSets, MyDataSets_Subset_4_9
from torchvision.utils import make_grid
from MyDataSet import MyDataSets_Subset_4_9
from matplotlib import pyplot as plt
import numpy as np
import torch


# pick_model = model.VaeFinal_only_one_hidden_copy()
# pick_model = model_probit.latent_space()

def print_counterfactuals(model, PRE_COUNTERFACTUAL_IMAGES, title=''):
    test_images_org_copy = PRE_COUNTERFACTUAL_IMAGES.clone()
    reconstructions = model(test_images_org_copy)[0]
    with torch.no_grad():
        print("Reconstructions")
        reconstructions = reconstructions.view(reconstructions.size(0), 1, 28, 28)
        reconstructions = reconstructions.cpu()
        reconstructions = reconstructions.clamp(0, 1)
        reconstructions = reconstructions[:50]
        plt.imshow(np.transpose(make_grid(reconstructions, 10, 5).numpy(), (1, 2, 0)))
        plt.title(f'{title}')
        plt.show()


def print_some_elements_return_them(model):
    test_images_org, _ = mydatasets_subset_4_9.dataloader_test_subset_one_batch()
    test_images_copy = test_images_org.clone()
    with torch.no_grad():
        print("Original Images")
        test_images = test_images_copy.cpu()
        test_images = test_images.clamp(0, 1)
        test_images = test_images[:50]
        test_images = make_grid(test_images, 10, 5)
        test_images = test_images.numpy()
        test_images = np.transpose(test_images, (1, 2, 0))
        plt.imshow(test_images)
        plt.show()
        return test_images_copy
        # print_counterfactuals(model=model, test_images_org=test_images_org)


def show_scatter(model, batch_from_dataloader_iter, current_epoch='epoch_information'):
    # helper.show_scatter_binary_dataset(model=model, mydataset_subset=mydataset, current_epoch=current_epoch)
    helper.show_scatter_with_batch_from_iter(model=model, batch_from_dataloader_iter=batch_from_dataloader_iter,
                                             current_epoch=current_epoch)
    # helper.


def set_params(BATCH_SIZE: int = 16, EPOCHS: int = 200, LR_RATE=0.0001):
    return BATCH_SIZE, EPOCHS, LR_RATE


def set_debug_params(TEST_AFTER_EPOCH=11, COUNT_PRINTS=11, SHOW_SCATTER_EVERY=5):
    return TEST_AFTER_EPOCH, COUNT_PRINTS, SHOW_SCATTER_EVERY


SHOW_COUNTERFACTUAL_EVERY = 2


def set_model_params_2HIDDEN(LATTENT_SPACE=2, HIDDEN_1_LAYER=200, HIDDEN_2_LAYER=-99, LAYER_2_AS_IDENTITY=True):
    return LATTENT_SPACE, HIDDEN_1_LAYER, HIDDEN_2_LAYER, LAYER_2_AS_IDENTITY


def set_model_params(LATTENT_SPACE=2,
                     HIDDEN_1_LAYER=200
                     ):
    return LATTENT_SPACE, HIDDEN_1_LAYER


# Set params
LATTENT_SPACE, HIDDEN_1_LAYER = set_model_params()
BATCH_SIZE, EPOCHS, LR_RATE = set_params()
TEST_AFTER_EPOCH, COUNT_PRINTS, SHOW_SCATTER_EVERY = set_debug_params()


def check_lr_change(loss, loss_last_min, counter_no_change, LR_RATE):
    break_it = False
    if loss < loss_last_min:
        loss_last_min = loss
        counter_no_change = 0
        print('counter_no_change = 0')
    else:
        if counter_no_change == 20:
            print('counter_no_change == 20 => break')
            break_it = True
        if counter_no_change == 10:
            LR_RATE /= 10
            print('counter_no_change == 10 =>Changed LR_RATE /= 10')
        counter_no_change += 1
    return loss_last_min, counter_no_change, LR_RATE, break_it


pick_model = model_probit.latent_space()
RUN_SAVE_NAME = pick_model.__class__.__name__ + str('')

mydatasets_subset_4_9 = MyDataSets_Subset_4_9(batch_size_train=BATCH_SIZE)
pick_device = 'cpu'
DEVICE = torch.device(pick_device)  # alternative 'mps' - but no speedup...
modelNN = pick_model.to(DEVICE)

# PRE_COUNTERFACTUAL_IMAGES = print_some_elements_return_them(model=model)

# optimizer = torch.optim.Adam(modelNN.parameters(), lr=LR_RATE)
optimizer = torch.optim.SGD(modelNN.parameters(), lr=LR_RATE)

# def return_new
# dataset_test_4 = MyDataSet.MyDataSets_Subset_4(batch_size_train=BATCH_SIZE)
# dataset_test_9 = MyDataSet.MyDataSets_Subset_9(batch_size_train=BATCH_SIZE)
#
# dataset49 = MyDataSet.MyDataSets_Subset_4_9(batch_size_train=-1).dataloader_train_subset()

# model_copy = model.VaeFinal_only_one_hidden_copy()
# model_copy = helper.import_model_name_weights_copy(model_x=model_copy, activate_eval=True)

# latent = model_copy.encode(images)
# latent = latent[0]
# latent =latent[0]
#
# calc_loss = nn.CrossEntropyLoss(reduction='mean')
# labels_t = torch.t(labels).unsqueeze(dim=1)
# print(f'{labels_t = }')
#
# training_img_labels = torch.hstack((latent, labels_t))
# print(f'{training_img_labels = }')

# ===========================
import MyDataSet
import pandas as pd
import torch
model_copy = model.VaeFinal_only_one_hidden_copy()
model_copy = helper.import_model_name_weights_copy(model_x=model_copy, activate_eval=True)
# dataset_test_4 = MyDataSet.MyDataSets_Subset_4(batch_size_train=-1)
with torch.no_grad():
    dataset_49 = MyDataSet.MyDataSets_Subset_4_9(batch_size_train=-1)
    img_49_batch, label_49_batch = next(iter(dataset_49.train_loader_subset_changed_labels))
    rec49, mu49, sigma49 = model_copy(img_49_batch.clone())
    z = mu49
df = pd.DataFrame({'z0': rec49[:, 0], 'z1': rec49[:, 1], 'labels': label_49_batch})
mcd = MyDataSet.MyCustomDataset(df)
dataloader = torch.utils.data.DataLoader(dataset=mcd,batch_size=BATCH_SIZE)
# calc_loss = nn.CrossEntropyLoss(reduction='mean')
calc_loss = nn.BCELoss(reduction='mean')

# ===========================

loss_last_min, counter_no_change = 0, 0
for epoch in range(EPOCHS):
    # loop = tqdm(enumerate(training_img_labels))
    loop = tqdm(enumerate(dataloader))
    # loop = tqdm(enumerate(dataset49))

    for i, (z, labels) in loop:
        labels = labels.to(DEVICE)
        z = z.to(DEVICE)

        y_pred = modelNN(z)
        loss = calc_loss(y_pred.view(-1), labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss)

print('finish!!!')
# print_counterfactuals(model=model, PRE_COUNTERFACTUAL_IMAGES=PRE_COUNTERFACTUAL_IMAGES)

SAVE_NAME_MODEL = RUN_SAVE_NAME + '_weights'
PATH = '/Users/dominikocsofszki/PycharmProjects/mlp/data/weights/' + SAVE_NAME_MODEL

print(f'save weights at {PATH = }')
print(f'{PATH}')
torch.save(modelNN.state_dict(), PATH)
