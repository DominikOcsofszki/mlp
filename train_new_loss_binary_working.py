import os

import mlflow
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

import helper
import model
from MyDataSet import MyDataSets_Subset, MyDataSets

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
def show_scatter(model, batch_from_dataloader_iter, current_epoch='epoch_information'):
    # helper.show_scatter_binary_dataset(model=model, mydataset_subset=mydataset, current_epoch=current_epoch)
    helper.show_scatter_with_batch_from_iter(model=model, batch_from_dataloader_iter=batch_from_dataloader_iter, current_epoch=current_epoch)
    # helper.


def set_params(BATCH_SIZE: int = 2 ** 6, EPOCHS: int = 300, LR_RATE=3e-4):
    return BATCH_SIZE, EPOCHS, LR_RATE


def set_debug_params(TEST_AFTER_EPOCH=11, COUNT_PRINTS=11, SHOW_SCATTER_EVERY=10):
    return TEST_AFTER_EPOCH, COUNT_PRINTS, SHOW_SCATTER_EVERY


def set_model_params(LATTENT_SPACE=2, HIDDEN_1_LAYER=200, HIDDEN_2_LAYER=-99,LAYER_2_AS_IDENTITY = True):
    return LATTENT_SPACE, HIDDEN_1_LAYER, HIDDEN_2_LAYER,LAYER_2_AS_IDENTITY


##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
# Set params
LATTENT_SPACE, HIDDEN_1_LAYER, HIDDEN_2_LAYER,LAYER_2_AS_IDENTITY = set_model_params()
BATCH_SIZE, EPOCHS, LR_RATE = set_params()
TEST_AFTER_EPOCH, COUNT_PRINTS, SHOW_SCATTER_EVERY = set_debug_params()

# Pick Model
# pick_model = model.VaeFinal()
pick_model = model.VaeFinal_1(LATTENT_SPACE=LATTENT_SPACE, HIDDEN_1_LAYER=HIDDEN_1_LAYER, HIDDEN_2_LAYER=HIDDEN_2_LAYER,LAYER_2_AS_IDENTITY=LAYER_2_AS_IDENTITY)


issues_with_connecting(having_issues=False)
RUN_SAVE_NAME = pick_model.__class__.__name__ + str('')
with mlflow.start_run(run_name=RUN_SAVE_NAME):
    MyDataSets_Subset = MyDataSets_Subset(batch_size_train=BATCH_SIZE)
    # MYDATASET = MyDataSets_Subset(batch_size_train=BATCH_SIZE)
    # MYDATASET = MyDataSets(tuble=use_classes,batch_size_train=BATCH_SIZE)
    pick_device = 'cpu'
    DEVICE = torch.device(pick_device)  # alternative 'mps' - but no speedup...
    model = pick_model.to(DEVICE)
    mlflow_start_log_first(EPOCHS, LR_RATE, BATCH_SIZE, pick_device, model)

    trainloader = MyDataSets_Subset.dataloader_train_subset()
    testloader = MyDataSets_Subset.dataloader_test_subset_one_batch()

    # loss_arr = []
    # test_img_after_epochs = []
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
    # loss_fn = nn.BCELoss(reduction='sum')
    calc_loss = nn.BCELoss(reduction='mean')
    for epoch in range(EPOCHS):

        loop = tqdm(enumerate(trainloader))

        for i, (x, _) in loop:
            x = x.to(DEVICE).view(x.shape[0], 28 * 28)
            x_reconstruction, mu, sigma = model(x)

            reconstruction_loss = calc_loss(x_reconstruction, x)
            kl_div = -torch.mean(
                1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
            alpha = 0.5
            beta = 1 - alpha
            loss = alpha * reconstruction_loss + beta * kl_div  # TODO Could also change or add alpha,beta weighting!
            # print(f'{loss = }')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item(), loss_avg=loss.item() / BATCH_SIZE, kl_div=kl_div.item(),reconstruction_loss=reconstruction_loss.item())
            # loss_arr.append((loss.item()).mean())
            # loop.set_postfix(loss_avg=loss.item()/BATCH_SIZE)
        if epoch % SHOW_SCATTER_EVERY == 0:
            # show_scatter(mydataset=MYDATASET, current_epoch=str(epoch), model=model)
            show_scatter(batch_from_dataloader_iter=MyDataSets_Subset.dataloader_test_subset_one_batch(),
                         current_epoch=str(epoch)+' l,h1,h2: '+str(set_model_params()), model=model)
    print('finish!!!')

    SAVE_NAME_MODEL = RUN_SAVE_NAME + '_weights'
    PATH = '/Users/dominikocsofszki/PycharmProjects/mlp/data/weights/' + SAVE_NAME_MODEL

    print(f'save weights at {PATH = }')
    print(f'{PATH}')
    torch.save(model.state_dict(), PATH)
