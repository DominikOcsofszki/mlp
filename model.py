import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(4, 10, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        # self.fc1 = nn.Linear(10*28*28,10)
        self.fc1 = nn.Linear(10 * 28 * 28, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        # print(x.shape)
        x = self.relu(self.conv2(x))
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        # print(x.shape)
        return x


class MyModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(4 * 28 * 28, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class MyModel3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(4 * 28 * 28, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class MyModel4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(20 * 28 * 28, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        # print(x.shape)
        x = self.relu(self.conv1(x))
        # print(x.shape)
        x = self.relu(self.conv2(x))
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        return x


class MyModel5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 100, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(100, 20, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(20 * 28 * 28, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        # print(x.shape)
        x = self.relu(self.conv1(x))
        # print(x.shape)
        x = self.relu(self.conv2(x))
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        return x


class MyModel6(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 100, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(100, 20, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(20 * 28 * 28, 20)
        self.fc2 = nn.Linear(20, 10)

        self.softmax = nn.Softmax()

    def forward(self, x):
        # print(x.shape)
        x = self.relu(self.conv1(x))
        # print(x.shape)
        x = self.relu(self.conv2(x))
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        x = self.softmax(x)
        return x


class MyModel7(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(20 * 28 * 28, 20)
        self.fc2 = nn.Linear(20, 10)
        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.softmax(x)
        return x


class MyModel8(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(8 * 28 * 28, 20)
        self.fc1_5 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 10)
        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc1_5(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class MyModel9(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 28 * 28, 20)
        self.fc2 = nn.Linear(20, 10)
        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class MyModel10(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(20, 10, kernel_size=3, stride=2, padding=1)
        # self.conv3 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        # self.fc1 = nn.Linear(10 * 28 * 28, 20)
        self.fc1 = nn.Linear(10 * 14 * 14, 20)
        self.fc2 = nn.Linear(20, 10)
        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class MyModel11(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=3, stride=1)
        # self.conv3 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        # self.fc1 = nn.Linear(10 * 28 * 28, 20)
        self.fc1 = nn.Linear(30 * 18 * 18, 20)
        self.fc2 = nn.Linear(20, 10)
        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print(x.shape)
        # x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class MyModel12(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=7, stride=1)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=3, stride=1)
        # self.conv3 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        # self.fc1 = nn.Linear(10 * 28 * 28, 20)
        self.fc1 = nn.Linear(30 * 14 * 14, 20)
        self.fc2 = nn.Linear(20, 10)
        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print(x.shape)
        # x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class MyModel13(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=7, stride=1)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=3, stride=1)
        # self.conv3 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        # self.fc1 = nn.Linear(10 * 28 * 28, 20)
        self.fc1 = nn.Linear(30 * 14 * 14, 100)
        self.fc2 = nn.Linear(100, 10)
        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print(x.shape)
        # x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class MyModel14(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=7, stride=1)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=7, stride=1)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(20 * 20 * 20, 10)
        # self.fc2 = nn.Linear(10, 10)
        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        # x = self.fc2(x)
        x = self.softmax(x)
        return x


class MyModel15(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=7, stride=1)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(30 * 14 * 14, 75)
        self.fc1_5 = nn.Linear(75, 10)
        self.fc2 = nn.Linear(10, 10)
        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc1_5(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def print_me(self):
        # for x in self.__dict__.items().__module__:
        for x in self.modules():
            print(x)


class MyModelVar(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=7, stride=1)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(30 * 14 * 14, 75)
        self.fc1_5 = nn.Linear(75, 10)
        self.fc2 = nn.Linear(10, 10)
        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc1_5(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def print_me(self):
        # for x in self.__dict__.items().__module__:
        for x in self.modules():
            print(x)


class MyModelVar_relu(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=7, stride=1)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(30 * 14 * 14, 75)
        self.fc1_5 = nn.Linear(75, 10)
        self.fc2 = nn.Linear(10, 10)
        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc1_5(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(x)
        return x

    def print_me(self):
        # for x in self.__dict__.items().__module__:
        for x in self.modules():
            print(x)


class MyModelVar_Without_Lin(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=7, stride=1)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(30, 20, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(20, 10, kernel_size=3, stride=1)
        self.conv6 = nn.Conv2d(10, 10, kernel_size=3, stride=1)
        self.conv7 = nn.Conv2d(10, 10, kernel_size=3, stride=1)
        self.conv8 = nn.Conv2d(10, 10, kernel_size=3, stride=1)
        self.conv9 = nn.Conv2d(10, 10, kernel_size=3, stride=2)

        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        # print(x.shape)
        x = self.flatten(x)

        x = self.softmax(x)
        return x

    def print_me(self):
        # for x in self.__dict__.items().__module__:
        for x in self.modules():
            print(x)


class MyModelVar_Without_Lin_change(nn.Module):
    def __init__(self):
        super().__init__()
        self.PRINT_ME = True
        self.conv1 = nn.Conv2d(1, 10, kernel_size=7, stride=1)  # 28-6-6-2 = 14
        self.conv2 = nn.Conv2d(10, 20, kernel_size=7, stride=1)  #
        self.conv3 = nn.Conv2d(20, 30, kernel_size=3, stride=1)  # 14
        self.conv4 = nn.Conv2d(30, 20, kernel_size=3, stride=2)  # 14/2 = 7 - 2 = 5
        self.conv5 = nn.Conv2d(20, 10, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(10, 10, kernel_size=3, stride=2)

        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()

    def forward(self, x):
        self.print_x_shape(x)
        x = self.conv1(x)
        self.print_x_shape(x)
        x = self.conv2(x)
        self.print_x_shape(x)
        x = self.conv3(x)
        self.print_x_shape(x)
        x = self.conv4(x)
        self.print_x_shape(x)
        x = self.conv5(x)
        self.print_x_shape(x)
        x = self.conv6(x)
        self.print_x_shape(x)
        # print(x.shape)
        x = self.flatten(x)
        self.print_x_shape(x)
        x = self.softmax(x)
        self.print_x_shape(x)
        if self.PRINT_ME: print(x[0])
        self.PRINT_ME = False
        return x

    def print_x_shape(self, x):
        if self.PRINT_ME: print(x.shape)

    def print_me(self):
        # for x in self.__dict__.items().__module__:
        for x in self.modules():
            print(x)


class CNN_6_CONV(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=7, stride=1)  # 28-6-6-2 = 14
        self.conv2 = nn.Conv2d(10, 20, kernel_size=7, stride=1)  #
        self.conv3 = nn.Conv2d(20, 30, kernel_size=3, stride=1)  # 14
        self.conv4 = nn.Conv2d(30, 20, kernel_size=3, stride=2)  # 14/2 = 7 - 2 = 5
        self.conv5 = nn.Conv2d(20, 10, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(10, 10, kernel_size=3, stride=2)

        self.softmax = nn.Softmax(dim=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(10, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def forward_mine(self, x):
        self.print_x_shape(x)
        x = self.conv1(x)
        self.print_x_shape(x)
        x = self.conv2(x)
        self.print_x_shape(x)
        x = self.conv3(x)
        self.print_x_shape(x)
        x = self.conv4(x)
        self.print_x_shape(x)
        x = self.conv5(x)
        self.print_x_shape(x)
        x = self.conv6(x)
        self.print_x_shape(x)
        # print(x.shape)
        x = self.flatten(x)
        self.print_x_shape(x)
        x = self.softmax(x)
        self.print_x_shape(x)
        if self.PRINT_ME: print(x[0])
        self.PRINT_ME = False
        return x

    def print_x_shape(self, x):
        if self.PRINT_ME: print(x.shape)


class vae(nn.Module):
    def __init__(self):
        super().__init__()
        self.PRINT_ME = True

        self.conv1 = nn.Conv2d(1, 10, kernel_size=7)  # 28-6-6-2 = 14
        self.conv2 = nn.Conv2d(10, 20, kernel_size=7)  #
        self.conv3 = nn.Conv2d(20, 30, kernel_size=3)  # 14

        self.conv4 = nn.Conv2d(30, 20, kernel_size=3)  # 14/2 = 7 - 2 = 5
        self.conv5 = nn.Conv2d(20, 10, kernel_size=7)
        self.maxpool = nn.MaxPool2d(2, 2)
        # self.conv6 = nn.Conv2d(10, 1, kernel_size=6)

        self.softmax = nn.Softmax(dim=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(10, 100)
        self.fc2 = nn.Linear(100, 10)

        self.upsample1 = nn.ConvTranspose2d(in_channels=10, out_channels=10, kernel_size=7, stride=1)
        self.upsample2 = nn.ConvTranspose2d(10, 10, kernel_size=3, stride=1)
        self.upsample3 = nn.ConvTranspose2d(10, 10, kernel_size=3, stride=1)
        self.upsample4 = nn.ConvTranspose2d(10, 10, kernel_size=7, stride=1)
        self.upsample5 = nn.ConvTranspose2d(10, 10, kernel_size=7, stride=1)
        self.upsample6 = nn.ConvTranspose2d(10, 10, kernel_size=2, stride=1)
        self.lin1 = nn.Linear(10 * 28 * 28, 32)
        self.lin2 = nn.Linear(32, 10)

    def print_x_shape(self, x):
        if self.PRINT_ME: print(x.shape)

    def encode(self, x):
        self.print_x_shape(x)
        x = self.conv1(x)
        self.print_x_shape(x)
        x = self.conv2(x)
        self.print_x_shape(x)
        x = self.conv3(x)
        self.print_x_shape(x)
        x = self.conv4(x)
        self.print_x_shape(x)
        x = self.conv5(x)
        self.print_x_shape(x)
        # x = self.conv6(x)
        x = self.maxpool(x)
        self.print_x_shape(x)
        x = self.maxpool(x)
        # self.print_x_shape(x)
        return x

    def decode(self, x):
        if self.PRINT_ME: print('--in-decoder:-----')

        # self.print_x_shape(x)
        x = self.upsample1(x)
        self.print_x_shape(x)
        x = self.upsample2(x)
        self.print_x_shape(x)
        x = self.upsample3(x)
        self.print_x_shape(x)
        x = self.upsample3(x)
        self.print_x_shape(x)
        x = self.upsample3(x)
        self.print_x_shape(x)
        x = self.upsample4(x)
        self.print_x_shape(x)
        x = self.upsample5(x)
        self.print_x_shape(x)
        x = self.upsample6(x)
        self.print_x_shape(x)
        if self.PRINT_ME: print('--upsampling-finished:-----')

        if self.PRINT_ME: print('--flattening:-----')

        if self.PRINT_ME: print('--in-decoder-finished:-----')

        return x

    def forward(self, x):
        if self.PRINT_ME: print('-----encoder:-----')
        x = self.encode(x)
        self.print_x_shape(x)
        if self.PRINT_ME: print('-----decoder:-----')
        x = self.decode(x)
        self.print_x_shape(x)

        if self.PRINT_ME: print('-----decoder-finished:-----')
        x = self.flatten(x)
        self.print_x_shape(x)
        x = self.lin1(x)
        self.print_x_shape(x)
        x = self.lin2(x)
        self.print_x_shape(x)
        x = self.softmax(x)
        self.print_x_shape(x)

        self.PRINT_ME = False
        return x


class vae2(nn.Module):
    def __init__(self):
        super().__init__()
        self.PRINT_ME = True

        self.conv1 = nn.Conv2d(1, 10, kernel_size=7)  # 28-6-6-2 = 14
        self.conv2 = nn.Conv2d(10, 20, kernel_size=7)  #
        self.conv3 = nn.Conv2d(20, 30, kernel_size=3)  # 14

        self.conv4 = nn.Conv2d(30, 20, kernel_size=3)  # 14/2 = 7 - 2 = 5
        self.conv5 = nn.Conv2d(20, 10, kernel_size=7)
        self.maxpool = nn.MaxPool2d(2, 2)
        # self.conv6 = nn.Conv2d(10, 1, kernel_size=6)

        self.softmax = nn.Softmax(dim=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(10, 100)
        self.fc2 = nn.Linear(100, 10)

        self.upsample1 = nn.ConvTranspose2d(in_channels=10, out_channels=10, kernel_size=7, stride=1)
        self.upsample2 = nn.ConvTranspose2d(10, 10, kernel_size=3, stride=1)
        self.upsample3 = nn.ConvTranspose2d(10, 10, kernel_size=3, stride=1)
        self.upsample4 = nn.ConvTranspose2d(10, 10, kernel_size=7, stride=1)
        self.upsample5 = nn.ConvTranspose2d(10, 10, kernel_size=7, stride=1)
        self.upsample6 = nn.ConvTranspose2d(10, 10, kernel_size=2, stride=1)
        self.lin1 = nn.Linear(10 * 28 * 28, 32)
        self.lin2 = nn.Linear(32, 10)

    def print_x_shape(self, x):
        if self.PRINT_ME: print(x.shape)

    def encode(self, x):
        self.print_x_shape(x)
        x = self.conv1(x)
        self.print_x_shape(x)
        x = self.conv2(x)
        self.print_x_shape(x)
        x = self.conv3(x)
        self.print_x_shape(x)
        x = self.conv4(x)
        self.print_x_shape(x)
        x = self.conv5(x)
        self.print_x_shape(x)
        # x = self.maxpool(x)
        # self.print_x_shape(x)
        # x = self.maxpool(x)
        return x

    def decode(self, x):
        if self.PRINT_ME: print('--in-decoder:-----')

        # self.print_x_shape(x)
        x = self.upsample1(x)
        self.print_x_shape(x)
        x = self.upsample2(x)
        self.print_x_shape(x)
        x = self.upsample3(x)
        self.print_x_shape(x)
        # x = self.upsample3(x)
        # self.print_x_shape(x)
        # x = self.upsample3(x)
        # self.print_x_shape(x)
        x = self.upsample4(x)
        self.print_x_shape(x)
        x = self.upsample5(x)
        self.print_x_shape(x)
        # x = self.upsample6(x)
        # self.print_x_shape(x)
        if self.PRINT_ME: print('--upsampling-finished:-----')

        if self.PRINT_ME: print('--flattening:-----')

        if self.PRINT_ME: print('--in-decoder-finished:-----')

        return x

    def forward(self, x):
        if self.PRINT_ME: print('-----encoder:-----')
        x = self.encode(x)
        self.print_x_shape(x)
        if self.PRINT_ME: print('-----decoder:-----')
        x = self.decode(x)
        self.print_x_shape(x)

        if self.PRINT_ME: print('-----decoder-finished:-----')
        x = self.flatten(x)
        self.print_x_shape(x)
        x = self.lin1(x)
        self.print_x_shape(x)
        x = self.lin2(x)
        self.print_x_shape(x)
        x = self.softmax(x)
        self.print_x_shape(x)

        self.PRINT_ME = False
        return x


class VAE(nn.Module):
    def __init__(self, input_dim=28 * 28, h_dim=200, z_dim=20):
        super().__init__()
        # encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        # decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)
        self.relu = nn.ReLU()

    def encode(self, x):
        # q_phi(z|x)
        h = self.relu(self.img_2hid(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma

    def decode(self, z):
        # p_theta(x|z)
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))

    def forward(self, x):
        mu, sigma = self.encode(x)
        eps = torch.randn_like(sigma)
        z_reparametrized = mu + sigma * eps
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, sigma


class autoencoder(nn.Module):
    def __init__(self, input_dim=28 * 28, h_dim=200, z_dim=20):
        super().__init__()
        # encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        # decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=1)

    def encode(self, x):
        # q_phi(z|x)
        x = self.flatten(x)
        h = self.relu(self.img_2hid(x))
        # mu, sigma = self.hid_2mu(h),self.hid_2sigma(h)
        # return mu,sigma
        return h

    def decode(self, z):
        # p_theta(x|z)
        # h = self.relu(self.z_2hid(z))
        h = self.relu(self.hid_2img(z))
        return h
        # return torch.sigmoid(self.hid_2img(h))

    def forward(self, x):
        # print(f'{x.shape =}')
        h = self.encode(x)
        # print(f'{h.shape =}')
        x_reconstructed = self.decode(h)
        # print(f'{x_reconstructed.shape =}')

        return x_reconstructed


class autoencoder_h5_n(nn.Module):
    def __init__(self, input_dim=28 * 28, h_dim=5):
        super().__init__()
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=1)

    def encode(self, x):
        x = self.flatten(x)
        return self.relu(self.img_2hid(x))

    def decode(self, z):
        return self.relu(self.hid_2img(z))

    def forward(self, x):
        h = self.encode(x)
        return self.decode(h)


class autoencoder_h5_n_weights_fin_acc89(nn.Module):
    def __init__(self, input_dim=28 * 28, h_dim=5):
        super().__init__()
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=1)

    def encode(self, x):
        x = self.flatten(x)
        return self.relu(self.img_2hid(x))

    def decode(self, z):
        return self.relu(self.hid_2img(z))

    def forward(self, x):
        h = self.encode(x)
        return self.decode(h)


class VaeMe(nn.Module):
    def __init__(self, hidden_units=500, latent=2):        #From paper hidden_units = 500 /
        super().__init__()                      #no overiffiting of superflouse latent variables,
        self.criterion = nn.CrossEntropyLoss()  #Could be explained by regularizing nature of the variational bound
        self.flatten = nn.Flatten(start_dim=1)
        # encode:
        self.img_to_hiden = nn.Linear(28 * 28, hidden_units)
        self.hiden_to_mu = nn.Linear(hidden_units, latent)
        self.hiden_to_sigma = nn.Linear(hidden_units, latent)
        # decode
        self.latent_to_hiden = nn.Linear(latent, hidden_units)
        self.hiden_to_rec_img = nn.Linear(hidden_units, 28 * 28)

    def return_loss_criterion_optimizer(self, lr_rate):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr_rate)
        return criterion, optimizer

    def loss_calculated_plus_term(self, loss):
        return loss + 100

    def encode(self, x):
        x = self.flatten(x)
        x = self.img_to_hiden(x)
        mu = self.hiden_to_mu(x)
        sigma = self.hiden_to_sigma(x)
        return mu, sigma

    def decode(self,z):
        z = self.latent_to_hiden(z)
        z = self.hiden_to_rec_img(z)
        return z
    def forward(self,x):
        mu, sigma = self.encode(x)
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps
        x = self.decode(z)
        return x


