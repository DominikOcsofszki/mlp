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

        self.fc1 = nn.Linear(10,100)
        self.fc2 = nn.Linear(100,10)

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
        self.maxpool = nn.MaxPool2d(2,2)
        # self.conv6 = nn.Conv2d(10, 1, kernel_size=6)

        self.softmax = nn.Softmax(dim=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(10,100)
        self.fc2 = nn.Linear(100,10)

    def forward__old__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool(x)

        # x = self.conv6(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

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
        # x = self.conv6(x)
        x = self.maxpool(x)
        self.print_x_shape(x)
        x = self.maxpool(x)
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
