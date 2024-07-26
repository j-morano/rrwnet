from pathlib import Path
import argparse
import time

import numpy as np
import torch
from torch import nn
from skimage import io
from thop import profile, clever_format


from net_utils import UNetModule, to_torch_tensors, pad_images_unet


class UNet(nn.Module):
    def __init__(self, input_ch, output_ch, base_ch):
        super().__init__()
        self.first_u = UNetModule(input_ch, output_ch, base_ch)

    def forward(self, x):
        pred = self.first_u(x)
        return [pred]


class WNet(nn.Module):
    def __init__(self, input_ch, output_ch, base_ch):
        super().__init__()
        self.first_u = UNetModule(input_ch, output_ch, base_ch)
        self.second_u = UNetModule(output_ch, 2, base_ch)

    def forward(self, x):
        predictions = []

        pred_1 = self.first_u(x)
        predictions.append(pred_1)
        bv_logits = pred_1[:, 2:3, :, :]
        pred_1 = torch.sigmoid(pred_1)

        pred_2 = self.second_u(pred_1)
        predictions.append(torch.cat((pred_2, bv_logits), dim=1))

        return predictions


# PROPOSED MODEL (BEST)
class RRWNet(nn.Module):
    def __init__(self, input_ch, output_ch, base_ch):
        super().__init__()
        self.first_u = UNetModule(input_ch, output_ch, base_ch)
        self.second_u = UNetModule(output_ch, 2, base_ch)

    def forward(self, x):
        predictions = []

        pred_1 = self.first_u(x)
        predictions.append(pred_1)
        bv_logits = pred_1[:, 2:3, :, :]
        pred_1 = torch.sigmoid(pred_1)
        bv = pred_1[:, 2:3, :, :]

        pred_2 = self.second_u(pred_1)
        predictions.append(torch.cat((pred_2, bv_logits), dim=1))

        for _ in range(5):
            pred_2 = torch.sigmoid(pred_2)
            pred_2 = torch.cat((pred_2, bv), dim=1)
            pred_2 = self.second_u(pred_2)
            predictions.append(torch.cat((pred_2, bv_logits), dim=1))

        return predictions


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='rrwnet')
    parser.add_argument('-o', '--option', type=str, default='time')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    model_factory = {
        'unet': UNet,
        'wnet': WNet,
        'rrwnet': RRWNet,
    }

    model = model_factory[args.model](3, 3, 64)
    print('Model:', model.__class__.__name__)

    model.eval()

    if torch.cuda.is_available() and not args.cpu:
        model.cuda()
        print('>> Using GPU (CUDA)')
    else:
        model.cpu()
        print('>> Using CPU')

    images_path = '_Data/RITE/test/enhanced'

    fns = sorted(Path(images_path).glob('*_test_enhanced.png'))

    measured_times = []
    # while True:
    for i, fn in enumerate(fns + [fns[0]]):
        img = io.imread(fn) / 255.0
        imgs, paddings = pad_images_unet([img])
        img = imgs[0]
        padding = paddings[0]
        # padding format: ((top, bottom), (left, right), (0, 0))
        tensors = to_torch_tensors([img])
        tensor = tensors[0]
        if torch.cuda.is_available() and not args.cpu:
            tensor = tensor.cuda()
        else:
            tensor = tensor.cpu()
        tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            if args.option == 'flops':
                macs, params = profile(model, inputs=(tensor, ))
                macs, params = clever_format([macs, params], "%.3f")
                print(f'MACs: {macs}, Parameters: {params}')
                exit(0)
            start = time.time()
            pred = model(tensor)[-1]
            pred = torch.sigmoid(pred)
            pred = pred[:, :, padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1]]
            elapsed = time.time() - start
            print('Elapsed time:', elapsed)
            if i > 0:
                measured_times.append(elapsed)

    mean_time = np.mean(measured_times)
    std_time = np.std(measured_times)
    print(f'Average time: {mean_time:.6f} +- {std_time:.6f} seconds')
    # Same in milliseconds
    print(f'Average time: {mean_time*1000:.2f} +- {std_time*1000:.2f} milliseconds')

