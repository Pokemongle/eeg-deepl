import torch.nn as nn
import torch
import numpy as np


class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        # 卷积层1
        self.conv1 = nn.Sequential(
            # 1X800X64, 10X11
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 9), stride=(1, 3), padding=(0, 4)),
            nn.BatchNorm2d(64),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            # 1X800X32, 10X11
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1)),
            nn.BatchNorm2d(32),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            # 1X800X16, 10X11
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1)),
            nn.BatchNorm2d(16),  # BatchNormalization层
            nn.ReLU(inplace=True)  # ReLU激活函数
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 9), stride=(1, 2), padding=(0, 4)),
            nn.BatchNorm2d(32),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 13), stride=(1, 2), padding=(0, 6)),
            nn.BatchNorm2d(32),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(32),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
        )
        self.convspa = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(2, 2), stride=(1, 1)),
            nn.BatchNorm2d(128),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2, 2), stride=(1, 1)),
            nn.BatchNorm2d(128),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2, 2), stride=(1, 1)),
            nn.BatchNorm2d(128),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2, 2), stride=(1, 1)),
            nn.BatchNorm2d(128),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2, 2), stride=(1, 1)),
            nn.BatchNorm2d(128),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2, 2), stride=(1, 1)),
            nn.BatchNorm2d(128),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2, 2), stride=(1, 1)),
            nn.BatchNorm2d(128),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2, 2), stride=(1, 1)),
            nn.BatchNorm2d(128),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2, 2), stride=(1, 1)),
            nn.BatchNorm2d(128),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 2), stride=(1, 1)),
            nn.BatchNorm2d(128),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
            nn.BatchNorm2d(128),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.MaxPool2d(kernel_size=(1, 2)),  # 添加 1x2 的 MaxPooling 层
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),  # BatchNormalization层
            nn.ReLU(inplace=True)  # ReLU激活函数
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
            nn.BatchNorm2d(128),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.MaxPool2d(kernel_size=(1, 2)),  # 添加 1x2 的 MaxPooling 层
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),  # BatchNormalization层
            nn.ReLU(inplace=True)  # ReLU激活函数
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
            nn.BatchNorm2d(128),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.MaxPool2d(kernel_size=(1, 2)),  # 添加 1x2 的 MaxPooling 层
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),  # BatchNormalization层
            nn.ReLU(inplace=True)  # ReLU激活函数
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
            nn.BatchNorm2d(128),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.MaxPool2d(kernel_size=(1, 2)),  # 添加 1x2 的 MaxPooling 层
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),  # BatchNormalization层
            nn.ReLU(inplace=True)  # ReLU激活函数
        )
        self.fc = nn.Linear(64 * 25, 30)  # Input size is the output size of the last convolutional layer
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 前向传播过程
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = torch.reshape(x, (400, 96, 10, 11))
        x = self.convspa(x)
        x = torch.reshape(x, (1, 64, 1, 400))
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = torch.reshape(x, (x.size(0), -1))  # Flatten the output of the last convolutional layer
        x = self.fc(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    # 原始输入矩阵
    input_matrix = torch.randn(10, 11, 1, 1, 2400)

    # 重塑为 (110, 1, 1, 2400)
    reshaped_input = torch.reshape(input_matrix, (110, 1, 1, 2400))

    # 创建 MyNetwork 实例
    net = MyNetwork()

    # 将重塑后的输入传入网络进行处理
    output = net(reshaped_input)

    # 输出的形状
    print("Output shape:", output.shape)

    print(MyNetwork())
    # 将输出重塑为原始形状 (10, 11, 1, 16, 2400)
    # reshaped_output = torch.reshape(output, (10, 11, 1, 16, 800))
    print(torch.cuda.devic)