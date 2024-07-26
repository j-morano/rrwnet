import argparse
from pathlib import Path
import json
from os.path import join

import test_utils
import constants



parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true")
parser.add_argument("--recompute", action="store_true")
parser.add_argument("-d", "--dataset", default="RITE-test")
parser.add_argument("-t", "--test", type=str, default="mav")
args = parser.parse_args()


data_path = Path('_Evaluation_Data')

results_dir = Path('__results')
results_dir.mkdir(exist_ok=True)

if args.dataset == "RITE-test":
    dataset = constants.rite_test_dataset
    dataset.paths = {
        'gt_hu',
        'gt2_qureshi',
        'masks',
        'rrwnet',
        'unet',
        'unet/refined_AV',
        'chen',
        'chen/refined_AV',
        'karlsson',
        'karlsson/refined_AV',
        'galdran',
        'galdran/refined_AV',
    }
elif args.dataset == "LES-AV":
    dataset = constants.lesav_dataset
    dataset.paths = {
        'gt_orlando',
        'masks',
        'rrwnet',
        'unet',
        'unet/refined_AV',
        'galdran',
        'galdran/refined_AV',
    }
elif args.dataset == "HRF":
    dataset = constants.hrf_dataset
    dataset.paths = {
        'gt_chen',
        'gt2_hemelings',
        'masks',
        'rrwnet',
        'unet',
        'unet/refined_AV',
        'chen',
        'chen/refined_AV',
        'karlsson',
        'karlsson/refined_AV',
        'galdran',
        'galdran/refined_AV',
    }
else:
    raise ValueError

# add samples to dataset Namespace
dataset.samples = {}


samples_fn = data_path / f"{dataset.name}_samples.json"

if Path(samples_fn).exists() and not args.force:
    print("Loading samples from file")
    with open(samples_fn, "r") as f:
        dataset.samples = json.load(f)
else:
    print("Generating samples file")
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

    with open(samples_fn, "w") as f:
        json.dump(dataset.samples, f, indent=4)


save_fn = join(results_dir, f'{dataset.name}_{args.test}.json')
assert Path(save_fn).parent.exists(), Path(save_fn).parent

paths = dataset.paths
paths = { p for p in paths if 'masks' not in p and 'gt' not in p }

results = {
    'All': {},
    'Intersection': {},
}

if args.test == 'mav':
    test_func = test_utils.mav
elif args.test == 'save_all':
    test_func = test_utils.save_all
elif args.test == 'topo_mp':
    test_func = test_utils.topo_mp
else:
    raise ValueError


n_paths = 100

if 'All' in results.keys():
    for i, path in enumerate(paths):
        results['All'][path] = test_func(
            dataset,
            path,
            results_dir,
            gt_key=dataset.gt,
            n_paths=n_paths,
        )
if 'Intersection' in results.keys():
    for i, path in enumerate(paths):
        results['Intersection'][path] = test_func(
            dataset,
            path,
            results_dir,
            predicted_only=True,
            gt_key=dataset.gt,
            n_paths=n_paths,
        )

with open(save_fn, 'w') as f:
    json.dump(results, f, indent=4)
