import os

import mlflow
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

import MyDataSet
import helper
import model
from MyDataSet import MyDataSets_Subset, MyDataSets, MyDataSets_Subset_4_9
from torchvision.utils import make_grid
from MyDataSet import MyDataSets_Subset_4_9
from matplotlib import pyplot as plt
import numpy as np
import torch

pick_model = model.VaeFinal_only_one_hidden_copy()
# pick_model = model.VaeFinal_only_one_hidden_copy_leakyrelu_for_momentum()

##############################################################################################################
##############################################################################################################
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:8000/'  # TODO Use this for SQL \ Delete for using local
start_database_terminal = 'pg_ctl -D /Users/dominikocsofszki/PycharmProjects/mlp/sql/sql1 -l logfile start'
start_mlflow_server = 'mlflow server --backend-store-uri postgresql://mlflow@localhost/mlflow_db --default-artifact-root file:"/Users/dominikocsofszki/PycharmProjects/mlp/mlruns" -h 0.0.0.0 -p 8000'


##############################################################################################################
##############################################################################################################
def mlflow_start_log_first(EPOCHS, LR_RATE, BATCH_SIZE, pick_device, model):
    print(f'{BATCH_SIZE = }')
    mlflow.log_param('epochs', EPOCHS)
    mlflow.log_param('LR_RATE', LR_RATE)
    # mlflow.log_param('MOMENTUM', MOMENTUM)
    mlflow.log_param('batch_size', BATCH_SIZE)
    mlflow.log_param('pick_device', pick_device)
    mlflow.log_param('model_name_full', model.__class__)
    mlflow.log_param('model_name', model.__class__.__name__)


def issues_with_connecting(having_issues=False):
    if having_issues:
        print(f'connecting to: http://localhost:8000/')
        print('if not working: open terminal with conda env mlp(source activate mlp):')
        print(start_database_terminal)
        print(start_mlflow_server)


##############################################################################################################
##############################################################################################################
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


def set_params(BATCH_SIZE: int = 128, EPOCHS: int = 100, LR_RATE=0.0001):
    return BATCH_SIZE, EPOCHS, LR_RATE


def set_debug_params(TEST_AFTER_EPOCH=11, COUNT_PRINTS=11, SHOW_SCATTER_EVERY=6):
    return TEST_AFTER_EPOCH, COUNT_PRINTS, SHOW_SCATTER_EVERY


SHOW_COUNTERFACTUAL_EVERY = 10


def set_model_params_2HIDDEN(LATTENT_SPACE=2, HIDDEN_1_LAYER=200, HIDDEN_2_LAYER=-99, LAYER_2_AS_IDENTITY=True):
    return LATTENT_SPACE, HIDDEN_1_LAYER, HIDDEN_2_LAYER, LAYER_2_AS_IDENTITY


def set_model_params(LATTENT_SPACE=2,
                     HIDDEN_1_LAYER=200
                     ):
    return LATTENT_SPACE, HIDDEN_1_LAYER


##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
# Set params
LATTENT_SPACE, HIDDEN_1_LAYER = set_model_params()
# LATTENT_SPACE, HIDDEN_1_LAYER, HIDDEN_2_LAYER,LAYER_2_AS_IDENTITY = set_model_params_2HIDDEN()
BATCH_SIZE, EPOCHS, LR_RATE = set_params()
TEST_AFTER_EPOCH, COUNT_PRINTS, SHOW_SCATTER_EVERY = set_debug_params()

# Pick Model
# pick_model = model.VaeFinal()
# pick_model = model.VaeFinal_only_one_hidden(LATTENT_SPACE=LATTENT_SPACE, HIDDEN_1_LAYER=HIDDEN_1_LAYER, HIDDEN_2_LAYER=HIDDEN_2_LAYER,LAYER_2_AS_IDENTITY=LAYER_2_AS_IDENTITY)
# pick_model = model.VaeFinal_CNN(LATTENT_SPACE=LATTENT_SPACE, HIDDEN_1_LAYER=HIDDEN_1_LAYER)
# pick_model = model.MyModel5_retry()

issues_with_connecting(having_issues=True)
RUN_SAVE_NAME = pick_model.__class__.__name__ + str('')

use_2_classifier = True
counter_no_change = 0
loss_last_min = 999_999
with mlflow.start_run(run_name=RUN_SAVE_NAME):
    if use_2_classifier:
        ###ADDING 2. Classifier
        pick_2_classifier = model.MyModel5_retry_2classes_faster_4()
        # pick_2_classifier = model.MyModel5_retry_2classes_faster()
        model2_classifier = helper.import_model_name(model_x=pick_2_classifier, activate_eval=True)
        model2_classifier.eval()
        # calc_loss_reconstruction_labeling = nn.BCELoss(reduction='sum')
        # calc_loss_reconstruction_labeling = nn.CrossEntropyLoss(reduction='sum')
        calc_loss_reconstruction_labeling = nn.CrossEntropyLoss(reduction='mean')
        print(f'{calc_loss_reconstruction_labeling = }')

        #=====================#
        #Add more classifier
        # pick_3_classifier = model.MyModel5_retry_2classes_faster_3()
        # # pick_2_classifier = model.MyModel5_retry_2classes_faster()
        # model3_classifier = helper.import_model_name(model_x=pick_3_classifier, activate_eval=True)
        # model3_classifier.eval()
        #
        # pick_4_classifier = model.MyModel5_retry_2classes_faster_2()
        # # pick_2_classifier = model.MyModel5_retry_2classes_faster()
        # model4_classifier = helper.import_model_name(model_x=pick_4_classifier, activate_eval=True)
        # model4_classifier.eval()


        #=====================#

    mydatasets_subset_4_9 = MyDataSets_Subset_4_9(batch_size_train=BATCH_SIZE)
    pick_device = 'cpu'
    DEVICE = torch.device(pick_device)  # alternative 'mps' - but no speedup...
    model = pick_model.to(DEVICE)
    mlflow_start_log_first(EPOCHS, LR_RATE, BATCH_SIZE, pick_device, model)

    trainloader = mydatasets_subset_4_9.dataloader_train_subset()
    testloader = mydatasets_subset_4_9.dataloader_test_subset()
    PRE_COUNTERFACTUAL_IMAGES = print_some_elements_return_them(model=model)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=LR_RATE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR_RATE,momentum=0.9)    #TODO If using SGD we need loss/batch_size
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR_RATE)
    # calc_kl_div_loss = nn.KLDivLoss(reduction="batchmean")

    # calc_loss = nn.BCELoss(reduction='sum')
    # calc_loss = nn.L1Loss(reduction='sum')
    calc_loss = nn.L1Loss(reduction='sum')
    # calc_loss = nn.MSELoss(reduction='sum')
    count_in_epoch = 0
    TRAIN_VAE = True
    use_2_classifier = True
    use_2_classifier_after = 1
    reconst_label_loss=torch.tensor(0)

    nr_of_classifier = 1
    LABEL_FACTOR = 10
    REC_LOSS_FACTOR = 1
    # REC_LOSS_FACTOR = 1 / nr_of_classifier
    KL_DIV_FACTOR = 10

    # KL_DIV_FACTOR = 1 / BATCH_SIZE
    if TRAIN_VAE:
        for epoch in range(EPOCHS):
            count_in_loop = 0
            loop = tqdm(enumerate(trainloader))

            if epoch == use_2_classifier_after :
                use_2_classifier=True
                REC_LOSS_FACTOR = 1
                LABEL_FACTOR = 1

            for i, (x, y) in loop:
                # print(f'{y = }')
                x = x.to(DEVICE).view(x.shape[0], 28 * 28)
                x_reconstruction, mu, sigma = model(x)
                if use_2_classifier:
                    x_rec_formated28_28 = x_reconstruction.reshape(x_reconstruction.shape[0], 1, 28, 28)
                    x_reconstruction_labeled = model2_classifier(x_rec_formated28_28)
                    reconst_label_loss = calc_loss_reconstruction_labeling(x_reconstruction_labeled, y)

                    # x_reconstruction_labeled2 = model3_classifier(x_rec_formated28_28)
                    # reconst_label_loss_2 = calc_loss_reconstruction_labeling(x_reconstruction_labeled2, y)
                    #
                    # x_reconstruction_labeled3 = model4_classifier(x_rec_formated28_28)
                    # reconst_label_loss_3 = calc_loss_reconstruction_labeling(x_reconstruction_labeled3, y)
                    # reconst_label_loss=reconst_label_loss+reconst_label_loss_2+reconst_label_loss_3

                reconstruction_loss = calc_loss(x, x_reconstruction)
                kl_div = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
                # kl_div = calc_kl_div_loss(input = (sigma,mu), target = )
                # reconstruction_loss /=BATCH_SIZE
                # kl_div*=BATCH_SIZE
                # kl_div += 1
                # nr_of_classifier = 1
                # LABEL_FACTOR = 1
                # REC_LOSS_FACTOR = 1/nr_of_classifier * 100000
                # KL_DIV_FACTOR = 1/BATCH_SIZE

                # loss = 4 * reconstruction_loss + 4 * kl_div  # TODO Could also change or add alpha,beta weighting!
                loss = REC_LOSS_FACTOR * reconstruction_loss + KL_DIV_FACTOR * kl_div  # TODO Could also change or add alpha,beta weighting!

                ##TODO adding extra loss
                if use_2_classifier:
                    loss = loss + LABEL_FACTOR * reconst_label_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # print(f'\n{kl_div.item()*KL_DIV_FACTOR = },\n{reconstruction_loss.item()*REC_LOSS_FACTOR = },\n'
            #       f'{reconst_label_loss.item()*LABEL_FACTOR = }\nTotalLoss:{loss.item() = },\n')
            print(f'\n{kl_div.item()*KL_DIV_FACTOR = },\n{reconstruction_loss.item()*REC_LOSS_FACTOR = },\n'
                  f'{reconst_label_loss.item()*LABEL_FACTOR = }\nTotalLoss:{loss.item() = },\n')

            if loss < loss_last_min:
                loss_last_min = loss
                counter_no_change = 0
                print('counter_no_change = 0')
            else:
                if counter_no_change == 20:
                    print('counter_no_change == 20 => break')
                    break
                if counter_no_change == 10:
                    LR_RATE /= 10
                    print('counter_no_change == 10 =>Changed LR_RATE /= 10')
                counter_no_change += 1

            if epoch % SHOW_SCATTER_EVERY == 0:
                # show_scatter(mydataset=MYDATASET, current_epoch=str(epoch), model=model)
                show_scatter(batch_from_dataloader_iter=mydatasets_subset_4_9.dataloader_test_subset_one_batch(),
                             current_epoch=str(epoch) + '  l,h1: ' + str(set_model_params()), model=model)
            if epoch % SHOW_COUNTERFACTUAL_EVERY == 0:
                # print_some_elements_return_them(model=model)
                print_counterfactuals(model=model, PRE_COUNTERFACTUAL_IMAGES=PRE_COUNTERFACTUAL_IMAGES,
                                      title=f'(epoch,loss, kl_div,reconstruction_loss,reconst_label_loss)\n' +
                                            str(epoch) + ', ' + str(int(loss)) + ', ' + str(
                                          {float(kl_div.item())}) + ', \n' +
                                            str(float(reconstruction_loss.item())) + ', ' + str(
                                          float(reconst_label_loss.item()))
                                      )

    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
        # optimizer = torch.optim.SGD(model.parameters(), lr=LR_RATE)    #TODO If using SGD we need loss/batch_size
        # calc_loss = nn.BCELoss(reduction='sum')
        calc_loss = nn.BCELoss()
        # calc_loss = nn.MSELoss()
        for epoch in range(EPOCHS):
            count_in_loop = 0
            loop = tqdm(enumerate(trainloader))

            for i, (X, y) in loop:
                y = y.to(DEVICE)
                X = X.to(DEVICE)
                y_pred = model(X)
                # y_pred = y_pred.view(BATCH_SIZE)
                print(f'{y = }')
                print(f'{y_pred = }')
                # loss = calc_loss(y_pred.squeeze(dim=1), y.float())
                loss = calc_loss(y_pred, y)
                print(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss.item())

    print('finish!!!')
    print_counterfactuals(model=model, PRE_COUNTERFACTUAL_IMAGES=PRE_COUNTERFACTUAL_IMAGES)

    SAVE_NAME_MODEL = RUN_SAVE_NAME + '_weights'
    PATH = '/Users/dominikocsofszki/PycharmProjects/mlp/data/weights/' + SAVE_NAME_MODEL

    print(f'save weights at {PATH = }')
    print(f'{PATH}')
    torch.save(model.state_dict(), PATH)
