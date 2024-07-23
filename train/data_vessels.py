from pathlib import Path
import os
from os.path import join
import re
from collections import defaultdict

from torch.utils.data import Dataset
import numpy as np
from skimage import io

import config



def read_image(file):
    img = io.imread(file)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    return img


def read_file(file):
    if '.npy' in file:
        data = np.load(file)
    elif '.npz' in file:
        data = np.load(file)['data']
    else:
        data = read_image(file)
        data = data.astype(np.float32)
        if data.max() > 1.0:
            data /= 255.0
    return data


class Img2ImgDataset:

    def __init__(self, data):
        self.target_path = os.path.join(data['data_folder'], data['target']['path'])
        self.original_path = os.path.join(data['data_folder'], data['original']['path'])
        self.mask_path = os.path.join(data['data_folder'], data['mask']['path'])

        self.target_pattern = data['target']['pattern']
        self.original_pattern = data['original']['pattern']
        self.mask_pattern = data['mask']['pattern']

        self._make_dataset()

    def _make_dataset(self):
        target = re.compile(self.target_pattern)
        orig = re.compile(self.original_pattern)
        mask = re.compile(self.mask_pattern)

        number = re.compile('[0-9]+')

        self.targets = defaultdict(dict)
        self.origs = defaultdict(dict)
        self.masks = defaultdict(dict)

        paths = [self.target_path, self.original_path, self.mask_path]
        patterns = [target, orig, mask]
        data_dicts = [self.targets, self.origs, self.masks]

        for path, pattern, data_dict in zip(paths, patterns, data_dicts):
            for file_name in os.listdir(path):
                if config.dataset == 'RITE' or config.dataset.startswith('LES-AV'):
                    n = number.findall(file_name)
                    if n:
                        n = int(n[0])
                        if pattern.match(file_name):
                            data_dict[n] = file_name
                else:
                    if pattern.match(file_name):
                        data_dict[Path(file_name).stem] = file_name


class VesselsDataset(Dataset, Img2ImgDataset):

    def __init__(self, data, transform=None):
        Img2ImgDataset.__init__(self, data)
        self.transform = transform
        self.vessels = self.targets
        self.retinos = self.origs
        # This allows no continuous keys in the dict:
        self.indices = [n for n in self.retinos.keys()]

    def __len__(self):
        return len(self.retinos)

    def __getitem__(self, index):
        _index = index
        retino = self.retinos[_index]
        vessel = self.vessels[_index]
        mask = self.masks[_index]

        assert isinstance(retino, str)
        assert isinstance(vessel, str)
        assert isinstance(mask, str)
        r = read_file(join(self.original_path, retino))
        m = read_file(join(self.mask_path, mask))
        v = read_file(join(self.target_path, vessel))

        item = [r, v, m]
        if self.transform is not None:
            item = self.transform(item)
        return [_index, item]

