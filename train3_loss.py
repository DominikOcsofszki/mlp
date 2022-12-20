import os
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
import torch.nn as nn

import helper
import model
import mlflow
from tqdm import tqdm

# mlflow server --backend-store-uri postgresql://mlflow@localhost/mlflow_db --default-artifact-root file:"/Users/dominikocsofszki/PycharmProjects/mlp/mlruns" -h 0.0.0.0 -p 8000
# DEBUG PARAMS:
TEST_ONLY_LAST = False
TEST_AFTER_EPOCH = 11
COUNT_PRINTS = 30
EPOCHS = 20

#
LR_RATE = 3e-4
pick_model = model.Vae_500h_2l()
ADD_TEXT = ''
RUN_SAVE_NAME = pick_model.__class__.__name__ + str(ADD_TEXT)
print(f'{RUN_SAVE_NAME = }')
HAS_LOSS_FUNCTION = True  # TODO If model has loss function implemented
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:8000/'  # TODO Use this for SQL \ Delete for using local
mlflow.set_experiment("train_new_loss_function")

# with mlflow.start_run(run_name=pick_model.__class__.__name__):
with mlflow.start_run(run_name=RUN_SAVE_NAME):
    BATCH_SIZE = 2**6
    pick_device = 'cpu'
    DEVICE = torch.device(pick_device)  # alternative 'mps' - but no speedup...
    model = pick_model.to(DEVICE)

    mlflow.log_param('epochs', EPOCHS)
    mlflow.log_param('LR_RATE', LR_RATE)
    mlflow.log_param('batch_size', BATCH_SIZE)
    mlflow.log_param('pick_device', pick_device)
    model_entries = [entry for entry in model.modules()]  ##need to ignore other files!?!?<<<<<<<<
    print(f'{model_entries.__len__()}')
    mlflow.log_param('model_name_full', model.__class__)
    mlflow.log_param('model_name', model.__class__.__name__)

    trainset = datasets.MNIST(root='data/dataset', train=True, transform=transforms.ToTensor(), download=True)
    testset = datasets.MNIST(root='data/testset', train=False, transform=transforms.ToTensor(), download=True)
    # Filter for only two classes #TODO Not sure yet if it is needed
    mlflow.log_param('trainset.len', trainset.__len__())
    mlflow.log_param('test.len', testset.__len__())
    # Trainloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE)  # No shuffle for reproducibility
    testsetloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)

    criterion = nn.BCELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=LR_RATE)


    def loss_calc(X, img_reconstructed, mu, sigma):
        # img_reconstructed = img_reconstructed.reshape(X.shape[0], 28*28)
        reconstruction_loss = criterion(X.reshape(img_reconstructed.shape[0], 28*28), img_reconstructed)
        # reconstruction_loss = criterion(X, img_reconstructed.view(X.shape[0], 28*28))
        kl_div = -torch.sum(
            1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))  # TODO Search in paper #minus for torch?
        # print(f'{reconstruction_loss = }')
        # print(f'{kl_div = }')
        loss = reconstruction_loss + kl_div
        return loss

    # Test function
    def test(dataloader, model_test):
        model_test.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                pred_mat, mu_mat, sigma_mat = model_test(X)
                loss = loss_calc(X, pred_mat, mu_mat, sigma_mat)
                print(loss)
                test_loss += loss.item()
        num_batches = len(dataloader)
        test_loss /= num_batches
        print(f'{epoch = } Avg loss: {test_loss:>8f}\n')
        return (100 * correct), test_loss

    accuracy, avg_loss = 0, 0
    acc_arr = []

    # for epoch in range(EPOCHS):
    #     running_loss = 0.0
    #     loop = tqdm(enumerate(trainloader))
    #     for i, data in loop:  # index = 0 could be deleted
    #
    #         X, y = data
    #         X, y = X.to(DEVICE), y.to(DEVICE)
    #
    #         optimizer.zero_grad()
    #         img_reconstructed, mu, sigma = model(X)
    #         loss = loss_calc(X, img_reconstructed, mu, sigma)
    #         loss.backward()
    #         optimizer.step()

    for epoch in range(EPOCHS):
        loop = tqdm(enumerate(trainloader))
        for i, (x, _) in loop:
            x = x.to(DEVICE).view(x.shape[0], 28*28)  # TODO why?
            x_reconstruction, mu, sigma = model(x)
            reconstruction_loss = criterion(x_reconstruction, x)
            kl_div = -torch.sum(
                1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))  # TODO Search in paper #minus for torch?
            loss = reconstruction_loss + kl_div  # TODO Could also change or add alpha,beta weighting!
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # test(dataloader=testsetloader, model_test=model)
    mlflow.log_param('acc_arr', acc_arr)
    mlflow.log_param('accuracy', accuracy)
    mlflow.log_param('avg_loss', avg_loss)

    SAVE_NAME_MODEL = RUN_SAVE_NAME + '_weights'
    PATH = '/Users/dominikocsofszki/PycharmProjects/mlp/data/weights/' + SAVE_NAME_MODEL

    print(f'save weights at {PATH = }')
    print(f'{PATH = }')
    torch.save(model.state_dict(), PATH)
