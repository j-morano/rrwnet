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
from test_utils import compute_classification_metrics, get_aucs



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
