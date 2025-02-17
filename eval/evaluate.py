import argparse
from pathlib import Path
import json
from os.path import join

import test_utils



parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default='Unknown')
parser.add_argument('-p', '--pred_path', type=str, default=None, help='Path to the predicted images')
parser.add_argument('-g', '--gt_path', type=str, default=None, help='Path to the ground truth images')
parser.add_argument('-m', '--mask_path', type=str, default=None, help='Path to the ROI masks')
parser.add_argument('-t', '--test', type=str, default='mav', choices=test_utils.test_factory.keys())
parser.add_argument('-n', '--n_paths', type=int, default=100)
parser.add_argument('-s', '--shape', type=str, default=None, help='Shape of the images for the evaluation. Format: "heightxwidth"')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results')
parser.add_argument('--results_dir', type=str, default='__results')
parser.add_argument('--pixels', type=str, default='both', choices=['both', 'intersection', 'all'])
parser.add_argument('-v', '--verbose', action='store_true')
args = parser.parse_args()


results_dir = Path(args.results_dir)
results_dir.mkdir(exist_ok=True)


test_func = test_utils.test_factory[args.test]

dataset = argparse.Namespace(
    name=args.dataset,
    paths=set(),
    samples=dict(),
    shape=None,
)

if args.shape is not None:
    dataset.shape = tuple(map(int, args.shape.split('x')))  # type: ignore


# Add samples to dataset Namespace
dataset.samples = {}

for path in Path(args.pred_path).iterdir():
    dataset.samples[path.stem] = {
        'pred': str(path),
    }
for path in Path(args.gt_path).iterdir():
    if path.stem in dataset.samples.keys():
        dataset.samples[path.stem]['gt'] = str(path)

if args.mask_path is None:
    args.mask_path = Path(args.gt_path).parent / 'enhanced_masks'

for path in Path(args.mask_path).iterdir():
    if path.stem in dataset.samples.keys():
        dataset.samples[path.stem]['masks'] = str(path)

for sample in dataset.samples.values():
    assert len(sample.keys()) == 3

if args.verbose:
    print(json.dumps(dataset.samples, indent=4))


results = dict()
if args.pixels == 'both' or args.pixels == 'all':
    results['All'] = test_func(
        dataset,
        'pred',
        results_dir,
        gt_key='gt',
        n_paths=args.n_paths,
    )
if args.pixels == 'both' or args.pixels == 'intersection':
    results['Intersection'] = test_func(
        dataset,
        'pred',
        results_dir,
        predicted_only=True,
        gt_key='gt',
        n_paths=args.n_paths,
    )


save_fn = join(results_dir, f'{dataset.name}_{args.test}_{args.shape}.json')
if Path(save_fn).exists() and not args.overwrite:
    print('Results file already exists')
    answer = input('Do you want to overwrite it? [y/n] ')
    if answer.lower() != 'y':
        i = 1
        while Path(save_fn).exists():
            save_fn = join(results_dir, f'{dataset.name}_{args.test}_{args.shape}_{i}.json')
            i += 1
        print(f'Using {save_fn} as filename')

with open(save_fn, 'w') as f:
    json.dump(results, f, indent=4)
print(f'Results saved to {save_fn}')
