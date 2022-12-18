import os

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import model
import mlflow
import VAE
from tqdm import tqdm
import real_vae

# Variables
TEST_ONLY_LAST = False
MOMENTUM = 0.9
# LR_RATE = 2e-2

LR_RATE = 2e-3
BATCH_SIZE = 64
EPOCHS = 1
pick_device = 'cpu'
DEVICE = torch.device(pick_device)  # alternative 'mps' - but no speedup...

# model = model.MyModel15().to(DEVICE)
model = model.vae().to(DEVICE)
# print(model.print_me())
print(model.__class__.__name__)

# ------------------TRACKING------------------------
# command for starting:
# mlflow server --backend-store-uri postgresql://mlflow@localhost/mlflow_db --default-artifact-root file:"/Users/dominikocsofszki/PycharmProjects/mlp/mlruns" -h 0.0.0.0 -p 8000

os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:8000/'  # TODO Use this for SQL \ Delete for using local
mlflow.start_run(run_name=model.__class__.__name__)  # ToDo put to the end for using "with" time missing
mlflow.log_param('epochs', EPOCHS)
mlflow.log_param('LR_RATE', LR_RATE)
mlflow.log_param('MOMENTUM', MOMENTUM)
mlflow.log_param('batch_size', BATCH_SIZE)
mlflow.log_param('pick_device', pick_device)
model_entries = [entry for entry in model.modules()]  ##need to ignore other files!?!?<<<<<<<<
print(f'{model_entries.__len__()}')

# mlflow.log_param('model_entries', model_entries)    #TODO Problem since to long
mlflow.log_param('model_name_full', model.__class__)
mlflow.log_param('model_name', model.__class__.__name__)
# mlflow.log_param('model_entries_0', model_entries[0]) #TODO find way to log model
# mlflow.log_param('model_entries_1', model_entries[1])
# mlflow.log_param('model_entries_2', model_entries[2])
# ------------------TRACKING-----------------------


# Downloading the dataset
trainset = datasets.MNIST(root='data/dataset', train=True, transform=transforms.ToTensor(), download=True)
testset = datasets.MNIST(root='data/testset', transform=transforms.ToTensor(), download=True)
# Filter for only two classes #TODO Not sure yet if it is needed
# TODO Do we need to normalize the data, or is it already done?

# Trainloader
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE)  # No shuffle for reproducibility
testsetloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)

# Loss function and optimizer


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR_RATE, momentum=MOMENTUM)


# optimizer = optim.SGD(model.parameters(), lr=1e-3)


# Test function
def test(dataloader, model_test, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model_test.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model_test(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f'Test Error:  Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}\n')
    return (100 * correct), test_loss


accuracy, avg_loss = 0, 0
acc_arr = []

for epoch in range(EPOCHS):

    running_loss = 0.0
    loop = tqdm(enumerate(trainloader))
    for i, data in loop:  # index = 0 could be deleted

        X, y = data
        X, y = X.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    if not TEST_ONLY_LAST:
        print(f'{epoch +1 = }')
        accuracy, avg_loss = test(testsetloader, model, criterion)
        acc_arr.append(float(f'{accuracy:.2f}'))
    else:
        if epoch + 1 == 5:
            accuracy, avg_loss = test(testsetloader, model, criterion)
            mlflow.log_param('accuracy_5', accuracy)
            mlflow.log_param('avg_loss_5', avg_loss)
        if epoch + 1 == 10:
            accuracy, avg_loss = test(testsetloader, model, criterion)
            mlflow.log_param('accuracy_10', accuracy)
            mlflow.log_param('avg_loss_10', avg_loss)
        if epoch + 1 == 15:
            accuracy, avg_loss = test(testsetloader, model, criterion)
            mlflow.log_param('accuracy_15', accuracy)
            mlflow.log_param('avg_loss_15', avg_loss)
if TEST_ONLY_LAST:
    accuracy, avg_loss = test(testsetloader, model, criterion)

mlflow.log_param('acc_arr', acc_arr)
mlflow.log_param('accuracy', accuracy)
mlflow.log_param('avg_loss', avg_loss)

print('finish!!!')

SAVE_NAME_MODEL = model.__class__.__name__ + '_weights'
# PATH = '/Users/dominikocsofszki/PycharmProjects/mlp/data/weights/weights_training'
# PATH = '/Users/dominikocsofszki/PycharmProjects/mlp/data/weights/weights_model_classifier_soft'
PATH = '/Users/dominikocsofszki/PycharmProjects/mlp/data/weights/'+SAVE_NAME_MODEL

print(f'save weights at {PATH = }')
torch.save(model.state_dict(), PATH)
