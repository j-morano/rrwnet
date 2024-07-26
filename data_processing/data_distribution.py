from pathlib import Path
import glob
import numpy as np
import json

from skimage import io



ground_truths_fns = sorted(glob.glob('HRF_AVLabel_191219/train_karlsson_w1024/av3/*.png'))
masks_fns = sorted(glob.glob('HRF_AVLabel_191219/train_karlsson_w1024/masks/*.png'))

data_samples = {
    'background': [0, 0],
    'vessel': [0, 0],
    'artery': [0, 0],
    'vein': [0, 0],
    'crossing': [0, 0],
    'uncertain': [0, 0]
}
total = 0
for ground_truth_fn, mask_fn in zip(ground_truths_fns, masks_fns):
    print(ground_truth_fn, mask_fn)
    assert Path(ground_truth_fn).stem == Path(mask_fn).stem
    ground_truth = io.imread(ground_truth_fn)/255
    mask = io.imread(mask_fn)/255
    A = ground_truth[:, :, 0]
    V = ground_truth[:, :, 1]
    VT = ground_truth[:, :, 2]
    background = (1 - VT) * mask
    # print(np.unique(background))
    num_background = np.sum(background).astype(int)
    # Crossings
    crossings = A * V
    num_crossings = np.sum(crossings).astype(int)
    # A/V
    A_nc = (A - crossings) * mask
    V_nc = (V - crossings) * mask
    # print(np.unique(A_nc), np.unique(V_nc))
    num_A = np.sum(A_nc).astype(int)
    num_V = np.sum(V_nc).astype(int)
    # Uncertain
    uncertain = VT - A - V
    uncertain[uncertain < 0] = 0
    # uncertain += crossings
    uncertain *= mask
    print(np.unique(uncertain))
    num_uncertain = np.sum(uncertain).astype(int)
    # Update dict
    data_samples['background'][0] += num_background
    data_samples['vessel'][0] += num_A + num_V + num_crossings + num_uncertain
    data_samples['artery'][0] += num_A
    data_samples['vein'][0] += num_V
    data_samples['crossing'][0] += num_crossings
    data_samples['uncertain'][0] += num_uncertain
    total += num_background + num_A + num_V + num_crossings + num_uncertain

for k, v in data_samples.items():
    data_samples[k][1] = int(round((data_samples[k][0] / total) * 100, 2))
    data_samples[k][0] = int(data_samples[k][0])

with open('_samples_distribution.json', 'w', encoding='utf-8') as fp:
    json.dump(data_samples, fp, ensure_ascii=False, indent=4)

print(data_samples)
