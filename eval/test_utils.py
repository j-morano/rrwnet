from pathlib import Path
from argparse import Namespace
import random
import pickle
import lzma
from multiprocessing import Pool
from typing import Union

import numpy as np
from sklearn.metrics import confusion_matrix
from skimage import io
from skimage.morphology import binary_erosion, disk
from sklearn.metrics import auc as compute_auc
from sklearn.metrics import roc_curve, precision_recall_curve

from topo_metric import topo_metric



def compute_classification_metrics(
    ground_truth,
    prediction,
    mask=None,
    threshold=0.5,
):
    # 2-class confusion matrix values
    if mask is not None:
        ground_truth = ground_truth[mask]
        prediction = prediction[mask]
    # Ensure that the values are 0 or 1
    ground_truth = np.where(ground_truth > 0.5, 1, 0)
    prediction = np.where(prediction > threshold, 1, 0)
    try:
        tn, fp, fn, tp = confusion_matrix(ground_truth, prediction).ravel()
    except ValueError:
        print('Value error excepted')
        tn, fp, fn, tp = 1, 0, 0, 1

    # Metrics
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)

    return sensitivity, specificity, accuracy, f1


def get_vessel_tree_mask(ground_truth):
    """ Gets the vessel tree mask from a ground truth image ignoring unknown parts.
    It also returns unknown vessels and crossings.
    Args:
        ground_truth: ground truth with format [Artery, Vein, Vessel Tree]
    Returns:
        (mask, unknown, crossings):
            vessel tree mask (not including unknown vessels nor crossings),
            unknown vessels,
            crossings (at the same time artery and vein)
    """

    # Unknown vessels (it is not known if they are arteries or veins)
    unknown = ground_truth[:, :, 2] - ground_truth[:, :, 0] - ground_truth[:, :, 1]
    unknown[unknown < 0] = 0
    # Crossings: at the same time artery and vein
    crossings = ground_truth[:, :, 0] * ground_truth[:, :, 1] * ground_truth[:, :, 2]
    # All undefined areas: is not possible to classify pixels as arteries or veins
    unk = crossings + unknown

    # All vessels mask without including unknown parts
    mask = ground_truth[:, :, 2] - unk

    return mask, unknown, crossings


def get_aucs(gts, preds, masks=None):
    if masks is None:
        fpr, tpr, _ = roc_curve(gts.flatten(), preds.flatten())
    else:
        fpr, tpr, _ = roc_curve(gts[masks], preds[masks])
    auc_roc = compute_auc(fpr, tpr)
    if masks is None:
        precision, recall, _ = precision_recall_curve(gts.flatten(), preds.flatten())
    else:
        precision, recall, _ = precision_recall_curve(gts[masks], preds[masks])
    auc_pr = compute_auc(recall, precision)
    return auc_roc, auc_pr



def save_all(
    dataset: Namespace,
    # Format: {sample_id: {'predicted': str, 'groundtruth': str, 'mask': str}}
    model: str,
    save_path: Union[str, Path],
    predicted_only: bool = False,
    n_paths: int = 100,
    gt_key: str = 'gt',
    masks_key: str = 'masks',
) -> dict:
    print('> Running save_all')
    samples = dataset.samples
    samples_to_save = []
    i = 0
    for sample in samples.values():
        print('.', end='', flush=True)
        img = io.imread(sample[model]) / 255.0
        ground = np.round(io.imread(sample[gt_key]) / 255.0)
        # fov_mask = io.imread(sample[masks_key])
        fov_mask = io.imread(sample[masks_key])
        if 'RITE' not in dataset.name:
            fov_mask = fov_mask / 255.0
        if len(fov_mask.shape) == 3:
            fov_mask = np.mean(fov_mask, axis=2)
        fov_mask = fov_mask > 0.5
        mask, unknown, _ = get_vessel_tree_mask(ground)
        mask = mask > 0.5

        fov_mask[unknown == 1] = 0
        # Erode fov_mask
        fov_mask = binary_erosion(fov_mask, disk(3))

        # Give random values between 0 and 1 to the pixels where
        #   channels 0 and 1 are 0.
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if img[x, y, 0] == 0 and img[x, y, 1] == 0 and mask[x, y]:
                    rand = random.random()
                    img[x, y, 0] = rand
                    img[x, y, 1] = 1 - rand
        pred_indices = 1 - np.argmax(img[:, :, :2], axis=2)
        # Only for predicted vessels
        if predicted_only:
            mask = mask * img[:, :, 2] > 0.5

        sample = {
            'indices_pred': pred_indices[mask],
            'indices_gt': ground[:, :, 0][mask],
            'pred_a': img[:, :, 0][fov_mask],
            'pred_v': img[:, :, 1][fov_mask],
            'gt_a': ground[:, :, 0][fov_mask],
            'gt_v': ground[:, :, 1][fov_mask],
        }
        samples_to_save.append(sample)

        i += 1
    print()

    model_name = model.replace('/', '_')

    with lzma.open(Path(save_path, f'{dataset.name}_{model_name}_samples.xz'), 'wb') as f:
        pickle.dump(samples_to_save, f)

    return {}


def get_topo_a_v_metrics(samples, model, gt_key, n_paths):
    infs_a = []
    corrs_a = []
    infs_v = []
    corrs_v = []
    for sample in samples.values():
        print('.', end='', flush=True)
        img = io.imread(sample[model]) / 255.0
        ground = np.round(io.imread(sample[gt_key]) / 255.0)

        inf_a, _, corr_a = topo_metric(ground[:, :, 0], img[:, :, 0], 0.5, n_paths)
        infs_a.append(inf_a/n_paths)
        corrs_a.append(corr_a/n_paths)

        inf_v, _, corr_v = topo_metric(ground[:, :, 1], img[:, :, 1], 0.5, n_paths)
        infs_v.append(inf_v/n_paths)
        corrs_v.append(corr_v/n_paths)
    return {
        'inf_a': infs_a,
        'corr_a': corrs_a,
        'inf_v': infs_v,
        'corr_v': corrs_v,
    }


def split_dict(d, n):
    step = len(d) // n
    splits = []
    keys = list(d.keys())
    for i in range(0, len(d), step):
        splits.append({k: d[k] for k in keys[i:i+step]})
    return splits


class TopoAVMetrics:
    def __init__(self, model, gt_key, n_paths):
        self.model = model
        self.gt_key = gt_key
        self.n_paths = n_paths

    def __call__(self, samples):
        return get_topo_a_v_metrics(samples, self.model, self.gt_key, self.n_paths)


def topo_mp(
    dataset: Namespace,
    # Format: {sample_id: {'predicted': str, 'groundtruth': str, 'mask': str}}
    model: str,
    save_path: Union[str, Path],
    verbose: bool = True,
    predicted_only: bool = False,
    n_paths: int = 100,
    gt_key: str = 'gt',
    masks_key: str = 'masks',
    num_processes: int = 7,
) -> dict:
    samples = dataset.samples

    samples_splits = split_dict(samples, num_processes)

    fun = TopoAVMetrics(model, gt_key, n_paths)

    with Pool(num_processes) as p:
        results = p.map(fun, samples_splits)
    # Wait for all processes to finish
    p.close()
    p.join()

    all_results = {}
    for res in results:
        for k, v in res.items():
            if k not in all_results:
                all_results[k] = []
            all_results[k].extend(v)

    if verbose:
        print('=' * 40)
        print(model)
        print('-' * 40)
        print('Inf A:', all_results['inf_a'])
        print('Corr A:', all_results['corr_a'])
        print('Inf V:', all_results['inf_v'])
        print('Corr V:', all_results['corr_v'])
        print('-' * 40)
        print('=' * 40)

    return all_results


def mav(
    dataset: Namespace,
    # Format: {sample_id: {'predicted': str, 'groundtruth': str, 'mask': str}}
    model: str,
    save_path: Union[str, Path],
    verbose: bool = True,
    predicted_only: bool = False,
    n_paths: int = 100,
    gt_key: str = 'gt',
    masks_key: str = 'masks',
) -> dict:
    h, w = dataset.shape
    samples = dataset.samples
    shape = (len(dataset.ids), h,w)
    all_preds_a = np.zeros(shape, np.float32)
    all_preds_v = np.zeros(shape, np.float32)
    all_gts_v = np.zeros(shape, np.float32)
    all_preds = np.zeros(shape, np.float32)
    all_gts = np.zeros(shape, np.float32)
    all_masks = np.zeros(shape, bool)
    all_pred_vessels = np.zeros(shape, np.float32)
    all_gt_vessels = np.zeros(shape, np.float32)
    all_mask_vessels = np.zeros(shape, bool)
    i = 0
    infs_a = []
    corrs_a = []
    infs_v = []
    corrs_v = []
    infs_bv = []
    corrs_bv = []
    for sample in samples.values():
        print('.', end='', flush=True)
        img = io.imread(sample[model]) / 255.0
        ground = np.round(io.imread(sample[gt_key]) / 255.0)
        fov_mask = io.imread(sample[masks_key])
        if len(fov_mask.shape) == 3:
            fov_mask = np.mean(fov_mask, axis=2)
        fov_mask = fov_mask > 0.5
        mask, _, _ = get_vessel_tree_mask(ground)
        mask = mask > 0.5

        inf_a, _, corr_a = topo_metric(ground[:, :, 0], img[:, :, 0], 0.5, n_paths)
        infs_a.append(inf_a/n_paths)
        corrs_a.append(corr_a/n_paths)

        inf_v, _, corr_v = topo_metric(ground[:, :, 1], img[:, :, 1], 0.5, n_paths)
        infs_v.append(inf_v/n_paths)
        corrs_v.append(corr_v/n_paths)

        inf_bv, _, corr_bv = topo_metric(ground[:, :, 2], img[:, :, 2], 0.5, n_paths)
        infs_bv.append(inf_bv/n_paths)
        corrs_bv.append(corr_bv/n_paths)

        arteries = ground[:, :, 0]
        # Give random values between 0 and 1 to the pixels where
        #   channels 0 and 1 are 0.
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if img[x, y, 0] == 0 and img[x, y, 1] == 0 and mask[x, y]:
                    rand = random.random()
                    img[x, y, 0] = rand
                    img[x, y, 1] = 1 - rand
        pred_indices = 1 - np.argmax(img[:, :, :2], axis=2)
        # Only for predicted vessels
        if predicted_only:
            mask = mask * img[:, :, 2] > 0.5

        all_preds_a[i] = img[:, :, 0]
        all_preds_v[i] = img[:, :, 1]
        all_gts_v[i] = ground[:, :, 1]
        all_preds[i] = pred_indices
        all_gts[i] = arteries
        all_masks[i] = mask
        all_pred_vessels[i] = img[:, :, 2]
        all_gt_vessels[i] = ground[:, :, 2]
        all_mask_vessels[i] = fov_mask
        i += 1
    print()

    all_preds_a = all_preds_a.flatten()
    all_preds_v = all_preds_v.flatten()
    all_gts_v = all_gts_v.flatten()
    all_gts = all_gts.flatten()
    all_preds = all_preds.flatten()
    all_masks = all_masks.flatten()
    all_gt_vessels = all_gt_vessels.flatten()
    all_pred_vessels = all_pred_vessels.flatten()
    all_mask_vessels = all_mask_vessels.flatten()

    sens_av, spec_av, acc_av, f1_av = compute_classification_metrics(all_gts, all_preds, all_masks)

    auc_roc_a, auc_pr_a = get_aucs(all_gts, all_preds_a, all_mask_vessels)

    auc_roc_v, auc_pr_v = get_aucs(all_gts_v, all_preds_v, all_mask_vessels)

    sens_bv, spec_bv, acc_bv, f1_bv = compute_classification_metrics(all_gt_vessels, all_pred_vessels, all_mask_vessels)

    auc_roc_bv, auc_pr_bv = get_aucs(all_gt_vessels, all_pred_vessels, all_mask_vessels)

    inf_a = [np.mean(infs_a), np.std(infs_a)]
    corr_a = [np.mean(corrs_a), np.std(corrs_a)]
    inf_v = [np.mean(infs_v), np.std(infs_v)]
    corr_v = [np.mean(corrs_v), np.std(corrs_v)]
    inf_bv = [np.mean(infs_bv), np.std(infs_bv)]
    corr_bv = [np.mean(corrs_bv), np.std(corrs_bv)]

    if verbose:
        print('=' * 40)
        print(model)
        print('-' * 40)
        print('Sens AV:', sens_av)
        print('Spec AV:', spec_av)
        print('Acc AV:', acc_av)
        print('F1 AV:', f1_av)
        print('-' * 40)
        print('AUC ROC A:', auc_roc_a)
        print('AUC PR A:', auc_pr_a)
        print('-' * 40)
        print('AUC ROC V:', auc_roc_v)
        print('AUC PR V:', auc_pr_v)
        print('-' * 40)
        print('Sens BV:', sens_bv)
        print('Spec BV:', spec_bv)
        print('Acc BV:', acc_bv)
        print('AUC ROC BV:', auc_roc_bv)
        print('AUC PR BV:', auc_pr_bv)
        print('F1 BV:', f1_bv)
        print('-' * 40)
        print('Inf A:', inf_a)
        print('Corr A:', corr_a)
        print('Inf V:', inf_v)
        print('Corr V:', corr_v)
        print('Inf BV:', inf_bv)
        print('Corr BV:', corr_bv)
        print('=' * 40)

    partial_results = {
        'sens_av': sens_av,
        'spec_av': spec_av,
        'acc_av': acc_av,
        'auc_roc_a': auc_roc_a,
        'auc_pr_a': auc_pr_a,
        'auc_roc_v': auc_roc_v,
        'auc_pr_v': auc_pr_v,
        'sens_bv': sens_bv,
        'spec_bv': spec_bv,
        'acc_bv': acc_bv,
        'auc_roc_bv': auc_roc_bv,
        'auc_pr_bv': auc_pr_bv,
        'inf_a': inf_a,
        'corr_a': corr_a,
        'inf_v': inf_v,
        'corr_v': corr_v,
        'inf_bv': inf_bv,
        'corr_bv': corr_bv,
    }
    return partial_results
