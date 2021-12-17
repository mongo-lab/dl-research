from __future__ import print_function
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
import datetime


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

        # 预训练vgg16的特征提取层
        self.features = vgg16.features
        self.features[0] = nn.Conv2d(1, 64, 3, 1, 1)

        self.avgpool = vgg16.avgpool
        # 添加新的全连接层
        self.classifier = nn.Sequential(
            nn.Linear(25088, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # 防止过拟合
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10)
        )

    # 定义前向传播路径
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # x = x.view(-1, 25088)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# 输出网络结构


jn_vgg = JnVgg16()
jn_vgg.to(device)
# net = vgg16
print(jn_vgg)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(jn_vgg.parameters(), lr=0.05, momentum=0.9)
optimizer = optim.SGD(jn_vgg.parameters(), lr=0.01)


# train
def train(epoch):
    running_loss = 0.0
    start = datetime.datetime.now()
    total_step = len(trainloader)
    for step, data in enumerate(trainloader, 0):
        inputs, target = data

        optimizer.zero_grad()
        if 'cpu' != device.type:
            inputs = inputs.cuda()
            target = target.cuda()
        outputs = jn_vgg(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if step % 1000 == 999:
            end = datetime.datetime.now()

            interval = end - start

            print('Epoch: %d |step: %d / %d |train loss: %.3f |cost time: %d s' % (epoch + 1, step, total_step, running_loss / 1000, interval.seconds))
            running_loss = 0.0
            start = datetime.datetime.now()


def test(epoch):
    model = jn_vgg
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for l_data in testloader:
            images, labels = l_data
            if 'cpu' != device.type:
                images = images.cuda()
                labels = labels.cuda()
            l_outputs = model(images)
            _, predicted = torch.max(l_outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Epoch: %d |accuracy: %d %% ' % (epoch + 1, 100 * correct / total))


def jn_vgg16_run():
    for epoch in range(5):
        train(epoch)
        test(epoch)


    # torch.save(jn_vgg, './model/jn_vgg16')


jn_vgg16_run()