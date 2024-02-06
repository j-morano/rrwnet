from typing import List, Tuple
from pathlib import Path
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import io
from torchvision import utils as vutils
import numpy as np

from preprocessing import enhance_image



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
    def __init__(self, input_ch=64, output_ch=32, bias=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(input_ch, output_ch, 2, 2, bias=bias)
        self.conv_block = nn.Sequential(self.conv)

    def forward(self, x):
        return self.conv_block(x)


class UNetModule(nn.Module):
    def __init__(self, input_ch, output_ch, base_ch):
        super().__init__()

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

        return x


def get_unet_padding_np(np_image: np.ndarray, n_down=5) -> tuple:
    """ Calculates the necessary padding of an image to be processed by
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
    """ Applies UNet padding to a list of images in NumPy format.

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


class RRWNet(nn.Module):
    def __init__(self, input_ch=3, output_ch=3, base_ch=64, iterations=5):
        super().__init__()
        self.first_u = UNetModule(input_ch, output_ch, base_ch)
        self.second_u = UNetModule(output_ch, 2, base_ch)
        self.iterations = iterations

    def forward(self, x):
        predictions = []

        pred_1 = self.first_u(x)
        predictions.append(pred_1)
        bv_logits = pred_1[:, 2:3, :, :]
        pred_1 = torch.sigmoid(pred_1)
        bv = pred_1[:, 2:3, :, :]

        pred_2 = self.second_u(pred_1)
        predictions.append(torch.cat((pred_2, bv_logits), dim=1))

        for _ in range(self.iterations):
            pred_2 = torch.sigmoid(pred_2)
            pred_2 = torch.cat((pred_2, bv), dim=1)
            pred_2 = self.second_u(pred_2)
            predictions.append(torch.cat((pred_2, bv_logits), dim=1))

        return predictions



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get predictions from a model')
    parser.add_argument('--weights', type=str, required=True,
        help='Path to the model weights')
    parser.add_argument('--images-path', type=str, required=True,
        help='Path to the images')
    parser.add_argument('--masks-path', type=str, required=True,
        help='Path to the masks')
    parser.add_argument('--save-path', type=str, required=True,
        help='Path to save the predictions')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the images')
    args = parser.parse_args()

    model = RRWNet()

    print(f'Loading model from {args.weights}')
    model.load_state_dict(torch.load(args.weights), strict=True)
    model.eval()

    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    print(f'Creating save path {args.save_path}')
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True)

    print(f'Getting images and masks from {args.images_path} and {args.masks_path}')
    image_fns = sorted(Path(args.images_path).glob('*.png'))
    mask_fns = sorted(Path(args.masks_path).glob('*.png'))

    print('Processing images')
    for image_fn, mask_fn in zip(image_fns, mask_fns):
        print(f'  {image_fn.name}')
        assert Path(mask_fn).stem == Path(image_fn).stem
        if args.preprocess:
            print('    Preprocessing first')
            img, mask = enhance_image(image_fn, mask_fn)
        else:
            img = (io.imread(image_fn) / 255.0)[..., :3]
            mask = io.imread(mask_fn) * 1.0
        imgs, paddings = pad_images_unet([img, mask])
        img = imgs[0]
        padding = paddings[0]
        mask = imgs[1]
        mask = np.stack([mask,] * 3, axis=2)
        mask_padding = paddings[1]
        tensors = to_torch_tensors([img, mask])
        image_tensor = tensors[0]
        mask_tensor = tensors[1]
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        else:
            tensor = image_tensor.cpu()
        image_tensor = image_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0)
        with torch.no_grad():
            predictions = model(image_tensor)
            last_pred = predictions[-1]
            last_pred = torch.sigmoid(last_pred)
            last_pred[mask_tensor < 0.5] = 0
            last_pred = last_pred[:, :, padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1]]
            target_fn = save_path / Path(image_fn).name
            vutils.save_image(last_pred, target_fn)
