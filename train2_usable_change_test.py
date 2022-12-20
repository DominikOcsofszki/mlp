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
TEST_ONLY_LAST = True
TEST_AFTER_EPOCH = 11
COUNT_PRINTS = 5
EPOCHS = 300

#
LR_RATE = 3e-4
# pick_model = model.VaeMe_200_hidden()
pick_model = model.Vae_500h_2l_sigma()
ADD_TEXT = ''
RUN_SAVE_NAME = pick_model.__class__.__name__ + str(ADD_TEXT)
print(f'{RUN_SAVE_NAME = }')
HAS_LOSS_FUNCTION = True  # TODO If model has loss function implemented
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:8000/'  # TODO Use this for SQL \ Delete for using local
# with mlflow.start_run(run_name=pick_model.__class__.__name__):
with mlflow.start_run(run_name=RUN_SAVE_NAME):
    for x in pick_model.parameters() :
        print (x.shape)
    print(list(pick_model.parameters()))
    # MOMENTUM = 0.9
    # BATCH_SIZE = 32 * 2 ** 1
    BATCH_SIZE = 16 * 2**1

    pick_device = 'cpu'
    DEVICE = torch.device(pick_device)  # alternative 'mps' - but no speedup...
    model = pick_model.to(DEVICE)
    print('its needed:')

    print(60000/BATCH_SIZE)

    mlflow.log_param('epochs', EPOCHS)
    mlflow.log_param('LR_RATE', LR_RATE)
    # mlflow.log_param('MOMENTUM', MOMENTUM)
    mlflow.log_param('batch_size', BATCH_SIZE)
    mlflow.log_param('pick_device', pick_device)
    model_entries = [entry for entry in model.modules()]  ##need to ignore other files!?!?<<<<<<<<
    print(f'{model_entries.__len__()}')
    mlflow.log_param('model_name_full', model.__class__)
    mlflow.log_param('model_name', model.__class__.__name__)
    # mlflow.log_param('model_entries_0', model_entries[0]) #TODO find way to log model
    # mlflow.log_param('model_entries_1', model_entries[1])
    # mlflow.log_param('model_entries_2', model_entries[2])
    # ------------------TRACKING-----------------------

    # Downloading the dataset
    trainset = datasets.MNIST(root='data/dataset', train=True, transform=transforms.ToTensor(), download=True)
    testset = datasets.MNIST(root='data/testset', train=False, transform=transforms.ToTensor(), download=True) #TODO use train3!!!
    # assert False #TODO do not use testset is missing train=False! For comparing here

    # Filter for only two classes #TODO Not sure yet if it is needed

    # Trainloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE)  # No shuffle for reproducibility
    testsetloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)

    # Loss function and optimizer
    if HAS_LOSS_FUNCTION:
        criterion, optimizer = model.return_loss_criterion_optimizer(LR_RATE)


    else:
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
                loss = loss_fn(pred, y).item()
                if HAS_LOSS_FUNCTION:
                    loss = model.loss_calculated_plus_term(loss)
                test_loss += loss
                # test_loss += loss_fn(pred, y).item()
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
            COUNT_PRINTS = helper.print_sth_once_ret_new_count(loss, COUNT_PRINTS, add_text='before')
            if HAS_LOSS_FUNCTION:
                loss = model.loss_calculated_plus_term(loss)
            COUNT_PRINTS = helper.print_sth_once_ret_new_count(loss, COUNT_PRINTS,add_text='after')
            loss.backward()
            optimizer.step()
        if not TEST_ONLY_LAST:
            print(f'{epoch +1 = }')
            accuracy, avg_loss = test(testsetloader, model, criterion)
            acc_arr.append(float(f'{accuracy:.1f}'))
        else:
            if (epoch + 1) % TEST_AFTER_EPOCH == 0:
                accuracy, avg_loss = test(testsetloader, model, criterion)
                acc_arr.append(float(f'{accuracy:.1f}'))

    if TEST_ONLY_LAST:
        accuracy, avg_loss = test(testsetloader, model, criterion)

    mlflow.log_param('acc_arr', acc_arr)
    mlflow.log_param('accuracy', accuracy)
    mlflow.log_param('avg_loss', avg_loss)

    print('finish!!!')

    # SAVE_NAME_MODEL = model.__class__.__name__ + '_weights'
    SAVE_NAME_MODEL = RUN_SAVE_NAME + '_weights'
    # PATH = '/Users/dominikocsofszki/PycharmProjects/mlp/data/weights/weights_training'
    # PATH = '/Users/dominikocsofszki/PycharmProjects/mlp/data/weights/weights_model_classifier_soft'
    PATH = '/Users/dominikocsofszki/PycharmProjects/mlp/data/weights/' + SAVE_NAME_MODEL

    print(f'save weights at {PATH = }')
    print(f'{PATH = }')
    torch.save(model.state_dict(), PATH)
