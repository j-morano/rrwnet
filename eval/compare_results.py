import argparse
from pathlib import Path
import json
from os.path import join

import constants
import test_utils



parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", default="RITE-test")
parser.add_argument("-p", "--predictions_path", type=str, default=None)
parser.add_argument("-t", "--test", type=str, default="mav")
parser.add_argument("-n", "--n_paths", type=int, default=100)
parser.add_argument("--force", action="store_true")
parser.add_argument("--recompute", action="store_true")
parser.add_argument("--data_path", type=str, default="_Evaluation_Data")
parser.add_argument("--results_dir", type=str, default="__results")
parser.add_argument("--pixels", type=str, default="both", choices=["both", "intersection", "all"])
args = parser.parse_args()


data_path = Path(args.data_path)

results_dir = Path(args.results_dir)
results_dir.mkdir(exist_ok=True)

if args.dataset == "RITE-test":
    dataset = constants.rite_test_dataset
    dataset.paths = {
        'gt_hu',
        'masks',
    }
    dataset.gt = 'gt_hu'
    if args.predictions_path is None:
        dataset.paths = dataset.paths.union({
            'gt2_qureshi',
            'rrwnet',
            'unet',
            'unet/refined_AV',
            'chen',
            'chen/refined_AV',
            'karlsson',
            'karlsson/refined_AV',
            'galdran',
            'galdran/refined_AV',
        })
elif args.dataset == "LES-AV":
    dataset = constants.lesav_dataset
    dataset.paths = {
        'gt_orlando',
        'masks',
    }
    dataset.gt = 'gt_orlando'
    if args.predictions_path is None:
        dataset.paths = dataset.paths.union({
            'rrwnet',
            'unet',
            'unet/refined_AV',
            'galdran',
            'galdran/refined_AV',
        })
elif args.dataset == "HRF-test":
    dataset = constants.hrf_dataset
    dataset.paths = {
        'gt_chen',
        'masks',
    }
    dataset.gt = 'gt_chen'
    if args.predictions_path is None:
        dataset.paths = dataset.paths.union({
            'gt2_hemelings',
            'rrwnet',
            'unet',
            'unet/refined_AV',
            'chen',
            'chen/refined_AV',
            'karlsson',
            'karlsson/refined_AV',
            'galdran',
            'galdran/refined_AV',
        })
else:
    raise ValueError

if args.predictions_path is not None:
    dataset.paths.add(args.predictions_path)
    save_fn = join(results_dir, f'{dataset.name}_model_{args.test}.json')
else:
    save_fn = join(results_dir, f'{dataset.name}_{args.test}.json')
assert Path(save_fn).parent.exists(), Path(save_fn).parent

# add samples to dataset Namespace
dataset.samples = {}

for path in dataset.paths:
    full_path = data_path / dataset.name / path
    assert full_path.exists(), full_path
    for id_ in dataset.ids:
        sample = {
            path: str(full_path / dataset.pattern.format(id_=id_))
        }
        if id_ not in dataset.samples:
            dataset.samples[id_] = sample
        else:
            dataset.samples[id_].update(sample)


paths = dataset.paths
paths = { p for p in paths if 'masks' not in p and 'gt' not in p }

if args.pixels == 'both':
    results = {
        'All': {},
        'Intersection': {},
    }
elif args.pixels == 'all':
    results = {
        'All': {},
    }
elif args.pixels == 'intersection':
    results = {
        'Intersection': {},
    }
else:
    raise ValueError("Invalid value for --pixels")

if args.test == 'mav':
    test_func = test_utils.mav
elif args.test == 'save_all':
    test_func = test_utils.save_all
elif args.test == 'topo_mp':
    test_func = test_utils.topo_mp
else:
    raise ValueError


if 'All' in results.keys():
    for i, path in enumerate(paths):
        results['All'][path] = test_func(
            dataset,
            path,
            results_dir,
            gt_key=dataset.gt,
            n_paths=args.n_paths,
        )
if 'Intersection' in results.keys():
    for i, path in enumerate(paths):
        results['Intersection'][path] = test_func(
            dataset,
            path,
            results_dir,
            predicted_only=True,
            gt_key=dataset.gt,
            n_paths=args.n_paths,
        )

with open(save_fn, 'w') as f:
    json.dump(results, f, indent=4)
