from pathlib import Path
import argparse
import json

from transformations import to_torch_tensors, pad_images_unet

import torch
from skimage import io
from torchvision import utils as vutils
import numpy as np


import factories



parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weights', type=str, required=True)
parser.add_argument('-t', '--test_name', type=str)
parser.add_argument('-i', '--images_path', type=str, required=True)
parser.add_argument('-m', '--masks_path', type=str, required=True)
args = parser.parse_args()


if args.test_name is None:
    if 'RITE' in args.images_path:
        if 'test' in args.images_path:
            args.test_name = 'RITE-test'
    elif 'LES-AV' in args.images_path:
        args.test_name = 'LES-AV'
    elif 'HRF' in args.images_path:
        if 'test' in args.images_path:
            args.test_name = 'HRF-test'

print('Loading model')
checkpoint = torch.load(Path(args.weights, 'generator_best.pth'))

print('Loading config')
with open(Path(args.weights, 'config.json'), 'r') as f:
    config = json.load(f)

print('Config:', config)
print(json.dumps(config, indent=4))

# Namespace fron config dict
config = argparse.Namespace(**config)

print('Creating model')
model = factories.ModelFactory().create_class(
    config.model,
    config.in_channels,
    config.out_channels,
    config.base_channels,
    config.num_iterations
)

print('Loading weights')
model.load_state_dict(checkpoint)

if torch.cuda.is_available():
    model.cuda()

save_path = Path(args.weights) / 'tests_predictions'
if args.test_name is not None:
    save_path = save_path / args.test_name
save_path.mkdir(exist_ok=True, parents=True)

for image_fn in Path(args.images_path).iterdir():
    mask_fn = None
    for mask_fn in Path(args.masks_path).iterdir():
        if mask_fn.stem == image_fn.stem:
            break
    if mask_fn is None:
        print(f'Mask not found for {image_fn}')
        exit(1)
    if image_fn.is_file():
        image = io.imread(image_fn)
        if image.max() > 1:
            image = image / 255.0
        mask = io.imread(mask_fn)
        if mask.max() > 1:
            mask = mask / 255.0
        images, paddings = pad_images_unet([image, mask], return_paddings=True)
        print('Paddings:', paddings)
        img = images[0]
        padding = paddings[0]
        mask = images[1]
        mask = np.stack([mask,] * 3, axis=2)
        mask_padding = paddings[1]
        # padding format: ((top, bottom), (left, right), (0, 0))
        print(img.shape, padding)
        tensors = to_torch_tensors([img, mask])
        tensor = tensors[0]
        mask_tensor = tensors[1]
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        else:
            tensor = tensor.cpu()
        tensor = tensor.unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0)
        with torch.no_grad():
            preds = model(tensor)
            if isinstance(preds, list):
                pred = preds[-1]
            else:
                pred = preds
            pred = torch.sigmoid(pred)
            pred[mask_tensor < 0.5] = 0
            pred = pred[:, :, padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1]]
            print(pred.shape)
            target_fn = save_path / Path(image_fn).name
            vutils.save_image(pred, target_fn)
