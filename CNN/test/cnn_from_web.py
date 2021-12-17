import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
# 1 prepare dataset

batch_size = 64

# ToTensor  转换图片为张量
# Normalized an tensor image with mean and standard deviation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='./data/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='./data/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# 2 design model


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 卷积1操作
        # input: channel=1, out_channel=10, kernel_size5*5;
        # output: h*w=(28-5+0+1)/1*(28-5+0+1)/1=24*24
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)

        # 卷积2操作 接收池化后的conv1 input(h*w)=h*w/2=24/2=12
        # input: channel=10, out_channel=20, kernel_size5*5;
        # output: h*w=(12-5+0+1)/1*(12-5+0+1)/1=8*8
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)

        # 最大池化
        self.pooling = torch.nn.MaxPool2d(2)

        # 全连接 接收池化后的conv2 input(h*w)=h*w/2=8/2=4
        # input: 展开后为20*(4*4)=320
        # output: 10 手写字10个分类
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        # flatten data from (n,1,28,28) to (n, 784)
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)  # -1 此处自动算出的是320
        x = self.fc(x)

        return x


model = CNN()

# 3 construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# 4 training cycle forward, backward, update
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[epoch %d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %% ' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
# #
#
# import matplotlib.pyplot as plt
# fig,ax=plt.subplots(nrows=2, ncols=5,sharex=True, sharey=True)
# ax = ax.flatten()
# for i in range(10):
#     img = X_train[y_train ==i][0].reshape(28,28)
#     ax[i].imshow(img,cmap='Greys',interpolation='nearest')
# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()
