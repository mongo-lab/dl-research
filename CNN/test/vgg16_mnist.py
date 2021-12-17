from __future__ import print_function
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

show = ToPILImage()
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)
#
batchSize = 4

##load data
transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.FashionMNIST(root='./data/fashion/', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=0)

testset = torchvision.datasets.FashionMNIST(root='./data/fashion/', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=0)

# 导入预训练模型
vgg16 = torchvision.models.vgg16(pretrained=True)
# 打印vgg16结构
print(vgg16)


class JnVgg16(nn.Module):
    def __init__(self):
        super(JnVgg16, self).__init__()
        # 增加一个提取层,1通道替换原有vgg16 接收3通道
        self.pre = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        # 预训练vgg16的特征提取层
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool
        # 添加新的全连接层
        self.classifier = nn.Sequential(
            nn.Linear(25088, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # 防止过拟合
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 10)
        )

    # 定义前向传播路径
    def forward(self, x):
        x = self.pre(x)
        F.relu(x)
        x = self.features(x)

        x = self.avgpool(x)
        x = x.view(-1, 25088)

        x = self.classifier(x)
        return x


# 输出网络结构


net = JnVgg16()
net.to(device)
# net = vgg16
print(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)

# train
print("training begin")
for epoch in range(3):
    start = time.time()
    running_loss = 0
    for i, data in enumerate(trainloader, 0):
        image, label = data
        if 'cpu' != device.type:
            image = image.cuda()
            label = label.cuda()
        image = Variable(image)
        label = Variable(label)
        optimizer.zero_grad()
        if i % 10 == 9:
            print(image.shape)
        outputs = net(image)
        loss = criterion(outputs, label)

        loss.backward()
        optimizer.step()

        running_loss += loss.data

        if i % 100 == 99:
            end = time.time()
            print('[epoch %d,imgs %5d] loss: %.7f  time: %0.3f s' % (
                epoch + 1, (i + 1) * 16, running_loss / 100, (end - start)))
            start = time.time()
            running_loss = 0
print("finish training")

# test
net.eval()
correct = 0
total = 0
for data in testloader:
    images, labels = data
    if 'cpu' != device.type:
        image = image.cuda()
        label = label.cuda()
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('Accuracy of the network on the %d test images: %d %%' % (total, 100 * correct / total))
