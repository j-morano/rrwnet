import json
import argparse
import pickle
from multiprocessing import Pool
import time
import os
import lzma
from pathlib import Path
import socket

from scipy.stats import wilcoxon
import numpy as np
from sklearn.metrics import auc as compute_auc
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix



parser = argparse.ArgumentParser()
parser.add_argument('--option', default='topo')
parser.add_argument('--dataset', default='HRF')
parser.add_argument('--metric', default='accuracy')
parser.add_argument('--nproc', default=10, type=int)
parser.add_argument('--model', default='karlsson')
parser.add_argument('--repetitions', default=5000, type=int)
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--sota', default='karlsson')
args = parser.parse_args()


if socket.gethostname() == 'hemingway':
    print('-> Running on hemingway, overwriting some arguments')
    args.nproc = 5


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


def get_aucs(gts, preds, masks=None):
    # print(gts.shape, preds.shape)
    # print(np.unique(gts), np.unique(preds))
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


def append_or_create(dic, key, value):
    if key in dic:
        dic[key].append(value)
    else:
        dic[key] = [value]


def join_results_dict(results):
    new_results = {}
    for res in results:
        for metric, values in res.items():
            if metric not in new_results:
                new_results[metric] = []
            new_results[metric].extend(values)
    return new_results


class BootstrapTester:
    def __init__(self, name: str, data: list, sample_size: int):
        self.name = name
        self.data = data
        self.sample_size = sample_size

    def __call__(self, iterations):
        # Do this in order to avoid the same random seed in all processes
        np.random.seed((os.getpid() * int(time.time())) % 123456789)
        values = {}
        for i in range(iterations):
            print(i, end='-', flush=True)
            indices = np.random.choice(len(self.data), size=self.sample_size, replace=True)
            c_samples = [self.data[j] for j in indices]
            all_indices_pred = np.concatenate([sample['indices_pred'] for sample in c_samples])
            all_indices_gt = np.concatenate([sample['indices_gt'] for sample in c_samples])
            all_pred_a = np.concatenate([sample['pred_a'] for sample in c_samples])
            all_pred_v = np.concatenate([sample['pred_v'] for sample in c_samples])
            all_gt_a = np.concatenate([sample['gt_a'] for sample in c_samples])
            all_gt_v = np.concatenate([sample['gt_v'] for sample in c_samples])

            _sensitivity, _specificity, accuracy, _f1 = \
                compute_classification_metrics(all_indices_gt, all_indices_pred)
            auroc_a, aupr_a = get_aucs(all_gt_a, all_pred_a)
            auc_v, aupr_v = get_aucs(all_gt_v, all_pred_v)
            append_or_create(values, 'accuracy', accuracy)
            # append_or_create(values, 'f1', f1)
            append_or_create(values, 'auroc_a', auroc_a)
            append_or_create(values, 'aupr_a', aupr_a)
            append_or_create(values, 'auc_v', auc_v)
            append_or_create(values, 'aupr_v', aupr_v)
        return values


if args.option == 'sample-bootstrap-mp':
    # Sample bootstrap test multiprocessing

    res_fn = f'__results_revision/{args.dataset}_{args.model}_bootstrap_results.json'
    if os.path.exists(res_fn) and not args.overwrite:
        print('Results file already exists')
        exit(0)

    samples_fn = f'__results_revision/{args.dataset}_{args.model}_samples.xz'
    print(f'Loading samples for model {args.model}')
    with lzma.open(samples_fn, 'rb') as f:
        samples = pickle.load(f)

    num_repetitions = args.repetitions
    # 50% of the dataset
    sample_size = len(samples) // 2

    '''
        sample = {
            'indices_pred': pred_indices[mask],
            'indices_gt': ground[:, :, 0][mask],
            'pred_a': img[:, :, 0][fov_mask],
            'pred_v': img[:, :, 1][fov_mask],
            'gt_a': ground[:, :, 0][fov_mask],
            'gt_v': ground[:, :, 1][fov_mask],
        }
    '''

    bootstrap_tester = BootstrapTester(args.model, samples, sample_size)
    num_processes = args.nproc
    process_args = [num_repetitions // num_processes] * num_processes
    print('Starting processes')
    print('Args:', process_args)
    with Pool(num_processes) as p:
        boostrap_results = p.map(bootstrap_tester, process_args)

    '''Results format:
        [{metric: [values], ...}, ...]
    '''

    boostrap_results = join_results_dict(boostrap_results)

    with open(res_fn, 'w') as f:
        json.dump(boostrap_results, f)


if args.option == 'significance':
    # Significance test

    base_path = '/mnt/Data/SSHFS/msc_server/r2av_eval/__results_revision/'
    base_path = Path(base_path)

    ours_fn = base_path / f'{args.dataset}_rwnetav_bootstrap_results.json'
    with open(ours_fn, 'r') as f:
        ours = json.load(f)
    sota_fn = base_path / f'{args.dataset}_{args.sota}_bootstrap_results.json'
    with open(sota_fn, 'r') as f:
        sota = json.load(f)

    print()
    metrics = set(ours.keys()).intersection(set(sota.keys()))
    p_values = {}
    for metric in metrics:
        print(f'Metric: {metric}')
        # print(np.mean(ours[metric]), np.std(ours[metric]))
        # print(np.mean(sota[metric]), np.std(sota[metric]))
        t_stat, p_value = wilcoxon(ours[metric], sota[metric])
        print('P-value:', p_value/2)
        print()
        p_values[metric] = p_value

    # with open(f'__results_revision/{args.dataset}_p_values.json', 'w') as f:
    #     json.dump(p_values, f)
