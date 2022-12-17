import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import model
import mlflow

# Variables
BATCH_SIZE = 2**4
EPOCHS = 10
# Change device to mps / cpu
# pick_device = 'cpu'
pick_device = 'cpu'
DEVICE = torch.device(pick_device)
# Downloading the dataset
trainset = datasets.MNIST(root='data/dataset', train=True, transform=transforms.ToTensor(), download=True)
testset = datasets.MNIST(root='data/testset', transform=transforms.ToTensor(), download=True)
# mlflow.autolog()
# Filter for only two classes #TODO Not sure yet if it is needed
# TODO Do we need to normalize the data, or is it already done?

# Trainloader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testsetloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

# Loss function and optimizer
model = model.MyModel().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)


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
    mlflow.log_param('epochs', EPOCHS)
    return (100 * correct), test_loss

mlflow.start_run()
mlflow.log_param('epochs', EPOCHS)
mlflow.log_param('batch_size', BATCH_SIZE)
mlflow.log_param('pick_device', pick_device)

accuracy, avg_loss = 0,0
for epoch in range(EPOCHS):

    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):  # index = 0 could be deleted

        X, y = data
        X, y = X.to(DEVICE), y.to(DEVICE)

        # X = X.permute(0,1,2,3)
        # print(X.shape)
        optimizer.zero_grad()

        pred = model(X)
        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # if i % 200 == 199:
        #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
        #     running_loss = 0.0
    print(f'{epoch = }')
    accuracy, avg_loss = test(testsetloader, model, criterion)
mlflow.log_param('accuracy', accuracy)
mlflow.log_param('avg_loss', avg_loss)


print('finish!!!')
PATH = '/Users/dominikocsofszki/PycharmProjects/mlp/data/weights/weights_training'

print(f'save weights at {PATH = }')
torch.save(model.state_dict(), PATH)

