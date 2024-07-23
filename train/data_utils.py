import csv

import torch
from torch.utils.data.sampler import Sampler



def get_folds(images, num_folds):
    num_folds = num_folds
    images_per_fold = int(len(images) / num_folds)

    images_folds = []
    for i in range(num_folds):
        images_folds.append(images[i * images_per_fold:(i + 1) * images_per_fold])

    folds = []
    for i in range(len(images_folds)):
        current_fold = {
            'validation': images_folds[i],
            'training': sum([sl for j, sl in enumerate(images_folds) if j != i], [])
        }
        folds.append(current_fold)

    return folds


class SubsetSequentialSampler(Sampler):
    """ Samples elements sequentially from a given list of indices, always in the same order.

    Args:
        indices (list): a list of indices
    """
    def __init__(self, indices, data_source=None):
        super().__init__(data_source)
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class SubsetRandomSampler(Sampler):
    """ Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (list): a list of indices
    """

    def __init__(self, indices, data_source=None):
        super().__init__(data_source)
        self.indices = indices

    def __iter__(self):
        return (self.indices[i.item()] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def save_to_csv(data, filepath):
    """ Writes a given data into a .csv.

    """
    with open(filepath, 'a') as file:
        writer = csv.writer(file)
        writer.writerows(data)
