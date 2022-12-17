import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import model

# Variables
BATCH_SIZE = 64
EPOCHS = 3
# Downloading the dataset
trainset = datasets.MNIST(root='data/dataset', train=True, transform=transforms.ToTensor(), download=True)
testset = datasets.MNIST(root='data/testset', transform=transforms.ToTensor(), download=True)

# Filter for only two classes #TODO Not sure yet if it is needed
# TODO Do we need to normalize the data, or is it already done?

# Trainloader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testsetloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

# Loss function and optimizer
model = model.MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# Test function
def test(dataloader,model_test,loss_fn) :
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model_test.eval()
    test_loss, correct = 0,0
    with torch.no_grad() :
        for X, y in dataloader:
            pred = model_test(X)
            test_loss += loss_fn(pred,y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct/=size
    print(f'Test Error:  Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n')


for epoch in range(EPOCHS):
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):  # index = 0 could be deleted

        X, y = data
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
    test(testsetloader, model, criterion)

print('finish!!!')