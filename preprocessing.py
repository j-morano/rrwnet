from pathlib import Path
from os.path import join
import argparse

import scipy.ndimage as ndimage
import numpy as np
import skimage.io as io
import glob
from PIL import Image
from scipy.interpolate import interp1d
from skimage.morphology import disk



def crop_center(img, cropx, cropy):
    y, x = img.shape[0], img.shape[1]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]


def to_0_1(img):
    interp_fun = interp1d([img.min(), img.max()], [0.0, 1.0])
    return interp_fun(img)


def enhance_image(img, mask, int_format=False, disk_size=5):
    """Enhance an image using the method described in the paper.
    Args:
        img (np.ndarray): Image to enhance
        mask (np.ndarray): ROI mask of the image to enhance

    Returns:
        Enhanced image
    """
    # Read image and its corresponding mask
    if isinstance(img, str) or isinstance(img, Path):
        img = io.imread(img)[..., :3]
    if isinstance(mask, str) or isinstance(mask, Path):
        mask = io.imread(mask)

    if len(img.shape) == 3:
        if img.shape[2] > 3:
            img = img[:, :, :3]
    if len(mask.shape) == 3:
        mask = np.sum(mask[:, :, :3], axis=2)

    img = img / 255
    # mask = np.where(mask > (255//2), 255, 0)
    mask = np.where(mask > 0.5, 1, 0)

    # Copy original image
    img_copy = img.copy()
    # Convert to PIL format
    zoomed_image = Image.fromarray(np.uint8(img_copy*255))
    # Enlarge image
    zoomed_image = zoomed_image.resize(
        (int(img_copy.shape[1]*1.15), int(img_copy.shape[0]*1.15)),
        Image.BICUBIC
    )
    # To numpy array type
    zoomed_image = np.array(zoomed_image)
    # Crop image to original size (zoom result)
    zoomed_image = crop_center(zoomed_image, img_copy.shape[1],
                               img_copy.shape[0])
    # Convert image from 0-255 format to 0.0-1.0 format
    zoomed_image = zoomed_image / 255.0

    # Create circular kernel for mask erosion
    kernel = disk(disk_size)

    # Erode mask
    mask = ndimage.binary_erosion(mask, kernel)
    # Convert boolean array to float array
    mask = mask * 1.0  # type: ignore

    img_copy[mask < 1.0] = 0.0

    # Create RGB mask (same mask for all channels)
    mask = np.stack((mask, mask, mask), axis=2)

    composed_image = mask.copy()
    composed_image[mask == 1.0] = img_copy[mask == 1.0]
    composed_image[mask < 1.0] = zoomed_image[mask < 1.0]

    filtered_image = ndimage.gaussian_filter(composed_image, sigma=(10, 10, 0))

    subtracted_image = composed_image - filtered_image
    subtracted_image[mask < 1.] = 0.

    enhanced_image = subtracted_image/np.std(subtracted_image)
    enhanced_image = to_0_1(enhanced_image)
    enhanced_image[mask < 1.] = 0.

    mask = mask[:, :, 0]

    if int_format:
        enhanced_image *= 255
        enhanced_image = enhanced_image.astype(np.uint8)
        mask *= 255
        mask = mask.astype(np.uint8)

    return enhanced_image, mask


def enhance_images(image_names, mask_names, save_path):
    """ Enhances a list of images.
    Args:
        image_names (list): List of image names.
        mask_names (list): List of ROI mask names.
    """
    for image_name, mask_name in zip(image_names, mask_names):
        assert Path(image_name).name == Path(mask_name).name
        enhanced_image, eroded_mask = enhance_image(image_name, mask_name, int_format=True)

        io.imsave(join(save_path, 'images', Path(image_name).name), enhanced_image)
        io.imsave(
            join(save_path, 'masks', Path(mask_name).name),
            eroded_mask
        )


def main(images_path, masks_path, save_path):
    # Get images and masks
    image_names = glob.glob(join(images_path, '*.png'))
    mask_names = glob.glob(join(masks_path, '*.png'))
    # Sort
    image_names.sort()
    mask_names.sort()
    print(image_names)
    print(mask_names)
    enhance_images(image_names, mask_names, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess images')
    parser.add_argument('--images-path', type=str, required=True,
        help='Path to the images')
    parser.add_argument('--masks-path', type=str, required=True,
        help='Path to the masks')
    parser.add_argument('--save-path', type=str, required=True,
        help='Path to save the enhanced images')
    args = parser.parse_args()
    save_path = Path(args.save_path)

    print('Preprocessing')
    save_path.mkdir(exist_ok=True)
    (save_path / 'masks').mkdir(exist_ok=True)
    (save_path / 'images').mkdir(exist_ok=True)
    main(args.images_path, args.masks_path, args.save_path)
