# coding=utf-8
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
import os
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt

# 创建文件夹
if not os.path.exists('./img2'):
    os.mkdir('./img2')


# 转成图片
def to_img(x):
    out = x
    out = out.view(-1, 1, 28, 28)
    return out


batch_size = 128
num_epoch = 200
z_dimension = 50
# 图像预处理
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1,), (0.5,))
])
mnist = datasets.MNIST(
    root='./mnist/', train=True, transform=img_transform, download=True
)
dataloader = torch.utils.data.DataLoader(
    dataset=mnist, batch_size=batch_size, shuffle=True
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# 定义判别器 多层全连接网络
# sigmoid激活函数得到一个0到1之间的概率进行二分类。
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784, 512),  # 输入特征数为784，输出为512
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),  # 进行非线性映射
            nn.Linear(512, 256),  # 进行一个线性映射
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()

        )

    def forward(self, x):
        x = self.dis(x)
        return x


# 定义生成器 多层全连接网络
# 输入为 50维的0-1高斯分布
# tanh激活函数，得到-1到1
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(50, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.gen(x)
        return x


# 创建对象
D = discriminator()
G = generator()
D.to(device)
G.to(device)


# 图片展示
def show(path, name):
    print(path)
    img_array = plt.imread(path)
    plt.imshow(img_array)
    plt.title(name)
    plt.axis('off')
    plt.show()


# 使用二分类交叉熵函数
criterion = nn.BCELoss()
criterion.to(device)
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

# 训练逻辑
for epoch in range(num_epoch):
    tt = True
    for i, (img, _) in enumerate(dataloader):
        num_img = img.size(0)

        img = img.view(num_img, -1)
        real_img = Variable(img).to(device)
        real_label = Variable(torch.ones(num_img)).to(device)
        fake_label = Variable(torch.zeros(num_img)).to(device)

        ##### 训练判别器
        real_out = D(real_img)
        d_loss_real = criterion(real_out.squeeze(), real_label)
        real_scores = real_out
        # 生成噪声
        z = Variable(torch.randn(num_img, z_dimension)).to(device)
        # 生成器，生成图片
        fake_img = G(z)
        # 判别器判别
        fake_out = D(fake_img)
        d_loss_fake = criterion(fake_out.squeeze(), fake_label)
        fake_scores = fake_out
        # 损失函数和优化
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        ##### 训练生成器
        z = Variable(torch.randn(num_img, z_dimension)).to(device)  # 得到随机噪声
        fake_img = G(z)  # 随机噪声输入到生成器中，得到一副假的图片
        #
        output = D(fake_img)
        g_loss = criterion(output.squeeze(), real_label)
        # bp and optimize
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        if (i + 1) % 300 == 0:
            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
                  'D real: {:.6f},D fake: {:.6f}'.format(
                epoch, num_epoch, d_loss.data.item(), g_loss.data.item(),
                real_scores.data.mean(), fake_scores.data.mean()  # 打印的是真实图片的损失均值
            ))
        if epoch == 0 and i == len(dataloader) - 1:
            real_images = to_img(real_img.to(device).data)
            real_path = './img2/real_images.png'
            save_image(real_images, real_path)
            show(real_path, "real image")

        if (epoch % 50 == 49 and tt == True) or (epoch == 0 and (i == len(dataloader) - 1)):
            fake_images = to_img(fake_img.to(device).data)
            fake_path = './img2/fake_images-{}.png'.format(epoch + 1);
            save_image(fake_images, fake_path)
            show(fake_path, "fake image")
            tt = False

# 保存模型
torch.save(G.state_dict(), './generator.pth')
torch.save(D.state_dict(), './discriminator.pth')