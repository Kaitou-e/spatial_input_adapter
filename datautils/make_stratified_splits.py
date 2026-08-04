"""
Stratified splits for Indian, Pavia, and Houston datasets.
Each split is saved as a .mat file with the same format as the original, but with new TR and TE variables containing the train and test labels, respectively. 
A JSON summary file is also created for each split, containing metadata about the split and class distributions.

Strata are created by shuffling the points of each class and splitting according to the specified train_ratio, ensuring that each class is represented in both train and test sets.


For each dataset, the output .mat file is named in the format:
 If a class has 100 pixels, and the train_ratio is 0.8, then 80 samples will be in the train set and 20 samples will be in the test set for that class.
 if a class has 16 pixels, and the train_ratio is 0.8, then 12 samples will be in the train set and 4 samples will be in the test set for that class.

"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


DATASETS = {
    'Indian': 'IndianPine.mat',
    'Pavia': 'Pavia.mat',
    'Houston': 'Houston.mat',
}


def split_labels(labels, train_ratio, seed):
    rng = np.random.default_rng(seed)
    train = np.zeros_like(labels)
    test = np.zeros_like(labels)
    summary = {}

    for class_id in sorted(int(value) for value in np.unique(labels) if value != 0):
        points = np.argwhere(labels == class_id)
        rng.shuffle(points)
        total = points.shape[0]
        n_train = int(round(total * train_ratio))
        if total > 1:
            n_train = min(max(n_train, 1), total - 1)

        train_points = points[:n_train]
        test_points = points[n_train:]
        train[train_points[:, 0], train_points[:, 1]] = class_id
        test[test_points[:, 0], test_points[:, 1]] = class_id
        summary[str(class_id)] = {
            'total': int(total),
            'train': int(train_points.shape[0]),
            'test': int(test_points.shape[0]),
        }

    return train, test, summary


def make_split(dataset, args):
    source = Path(args.data_dir) / DATASETS[dataset]
    data = loadmat(source)
    labels = data['TR'] + data['TE']
    train, test, summary = split_labels(labels, args.train_ratio, args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_pct = int(round(args.train_ratio * 100))
    test_pct = 100 - train_pct
    output = output_dir / f'{dataset}_stratified_{train_pct}_{test_pct}_seed{args.seed}.mat'
    summary_output = output.with_suffix('.json')

    savemat(output, {'input': data['input'], 'TR': train, 'TE': test})
    payload = {
        'dataset': dataset,
        'source': str(source),
        'split': str(output),
        'train_ratio': args.train_ratio,
        'seed': args.seed,
        'classes': summary,
        'totals': {
            'train': int(np.count_nonzero(train)),
            'test': int(np.count_nonzero(test)),
        },
    }
    summary_output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')

    print(f'{dataset}: wrote {output}')
    print(f"  train={payload['totals']['train']} test={payload['totals']['test']}")
    print(f'  summary={summary_output}')


def main():
    ROOT = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser('Create one saved 80/20 stratified split per dataset')
    parser.add_argument('--dataset', choices=['all', *DATASETS], default='all')
    parser.add_argument('--data_dir', default=ROOT / 'data')
    parser.add_argument('--output_dir', default=ROOT / 'data' / 'splits')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    for dataset in datasets:
        make_split(dataset, args)


if __name__ == '__main__':
    main()
