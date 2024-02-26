from pathlib import Path
import argparse

import torch
from skimage import io
from torchvision import utils as vutils
import numpy as np

from preprocessing import enhance_image
from model import RRWNet
from utils import pad_images_unet, to_torch_tensors



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
    parser.add_argument('--refine', action='store_true', help='Refine the predictions')
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
            if args.refine:
                predictions = model.refine(image_tensor)
            else:
                predictions = model(image_tensor)
            last_pred = predictions[-1]
            if not args.refine:
                last_pred = torch.sigmoid(last_pred)
            last_pred[mask_tensor < 0.5] = 0
            last_pred = last_pred[:, :, padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1]]
            target_fn = save_path / Path(image_fn).name
            vutils.save_image(last_pred, target_fn)
