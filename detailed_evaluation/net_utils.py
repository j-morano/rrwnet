from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np



class ConvBlock(nn.Module):

    def __init__(self, input_ch=3, output_ch=64, activf=nn.ReLU, bias=True):
        super().__init__()

        self.conv1 = nn.Conv2d(input_ch, output_ch, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(output_ch, output_ch, 3, 1, 1, bias=bias)
        self.activf = activf

        self.conv_block = nn.Sequential(
            self.conv1,
            self.activf(inplace=True),
            self.conv2,
            self.activf(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


class UpConv(nn.Module):

    def __init__(self, input_ch=64, output_ch=32, activf=nn.ReLU, bias=True):
        super().__init__()

        self.conv = nn.ConvTranspose2d(input_ch, output_ch, 2, 2, bias=bias)
        self.conv_block = nn.Sequential(self.conv)

    def forward(self, x):
        return self.conv_block(x)


class UNetModule(nn.Module):

    def __init__(self, input_ch, output_ch, base_ch):
        super(UNetModule, self).__init__()

        self.conv1 = ConvBlock(input_ch, base_ch)
        self.conv2 = ConvBlock(base_ch, 2* base_ch)
        self.conv3 = ConvBlock(2 * base_ch, 4 * base_ch)
        self.conv4 = ConvBlock(4 * base_ch, 8 * base_ch)
        self.conv5 = ConvBlock(8 * base_ch, 16 * base_ch)

        self.upconv1 = UpConv(16 * base_ch, 8 * base_ch)
        self.conv6 = ConvBlock(16 * base_ch, 8 * base_ch)
        self.upconv2 = UpConv(8 * base_ch, 4 * base_ch)
        self.conv7 = ConvBlock(8 * base_ch, 4 * base_ch)
        self.upconv3 = UpConv(4 * base_ch, 2 * base_ch)
        self.conv8 = ConvBlock(4 * base_ch, 2 * base_ch)
        self.upconv4 = UpConv(2 * base_ch, base_ch)
        self.conv9 = ConvBlock(2 * base_ch, base_ch)

        self.outconv = nn.Conv2d(base_ch, output_ch, 1, bias=True)

    def forward(self, x):

        x1 = self.conv1(x)
        x = F.max_pool2d(x1, 2, 2)

        x2 = self.conv2(x)
        x = F.max_pool2d(x2, 2, 2)

        x3 = self.conv3(x)
        x = F.max_pool2d(x3, 2, 2)

        x4 = self.conv4(x)
        x = F.max_pool2d(x4, 2, 2)

        x = self.conv5(x)
        x = self.upconv1(x)
        x = torch.cat((x4, x), dim=1)

        x = self.conv6(x)
        x = self.upconv2(x)
        x = torch.cat((x3, x), dim=1)

        x = self.conv7(x)
        x = self.upconv3(x)
        x = torch.cat((x2, x), dim=1)

        x = self.conv8(x)
        x = self.upconv4(x)
        x = torch.cat((x1, x), dim=1)

        x = self.conv9(x)
        x = self.outconv(x)

        return x  # , internal


def get_unet_padding_np(np_image: np.ndarray, n_down=4) -> tuple:
    """Calculates the necessary padding of an image to be processed by
    UNet.

    Args:
        np_image: image in NumPy format.
        n_down: number of downsampling blocks.

    Returns:
        Image padding (NumPy format).
    """
    n = 2**n_down
    shape = np_image.shape
    h_pad = n - shape[0] % n
    w_pad = n - shape[1] % n
    h_half_pad = int(h_pad/2)
    w_half_pad = int(w_pad/2)
    if len(shape) == 3:
        padding = (h_half_pad, h_pad-h_half_pad), (w_half_pad, w_pad-w_half_pad), (0, 0)
    else:
        padding = (h_half_pad, h_pad-h_half_pad), (w_half_pad, w_pad-w_half_pad)
    return padding


def pad_images_unet(np_images: List[np.ndarray]) -> Tuple[List, List]:
    """Applies UNet padding to a list of images in NumPy format.

    Args:
        np_images: list of NumPy images.

    Returns:
        Padded images.
    """
    padded_images = []
    paddings = []
    for np_image in np_images:
        padding = get_unet_padding_np(np_image)
        paddings.append(padding)
        padded_images.append(np.pad(np_image, padding))
    return padded_images, paddings


def to_torch_tensors(npimages):
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    tensors = []
    for img in npimages:
        if len(img.shape) == 2:
            tensors.append(torch.from_numpy(img[np.newaxis, ...].astype('float32')))
        else:
            tensors.append(torch.from_numpy(img.transpose(2, 0, 1).astype('float32')))
    return tensors
