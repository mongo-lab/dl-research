import os
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
# from tools.common_tools import transform_invert, set_seed


# ================================= load img ==================================
path_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lena.jpg")
img = Image.open(path_img).convert('RGB')  # 0~255

# convert to tensor
img_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = img_transform(img)
img_tensor.unsqueeze_(dim=0)    # C*H*W to B*C*H*W

# ========================= create convolution layer ==========================
conv_layer = nn.Conv2d(3, 1, 3)   # input:(i, o, size) weights:(o, i , h, w)
nn.init.xavier_normal_(conv_layer.weight.data)

# calculation
img_conv = conv_layer(img_tensor)

# =========================== visualization ==================================
print("卷积前尺寸:{}\n卷积后尺寸:{}".format(img_tensor.shape, img_conv.shape))
# img_conv = transform_invert(img_conv[0, 0:1, ...], img_transform)
# img_raw = transform_invert(img_tensor.squeeze(), img_transform)
plt.subplot(122).imshow(img_transform, cmap='gray')
# plt.subplot(121).imshow(img_raw)
plt.show()



def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
# Get a batch of training data


def show(data_load):
    oo, classes = next(iter(data_load))
    # Make a grid from batch
    out = torchvision.utils.make_grid(_[0])
    imshow(out)
