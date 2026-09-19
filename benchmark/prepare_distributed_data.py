"""Freeze official MNIST and local road data, checksums and leakage-aware splits."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request

import cv2
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def prepare_mnist(out):
    files = {}
    arrays = {}
    for name in ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
                 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'):
        path = out / (name + '.gz')
        url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + path.name
        if not path.exists():
            print('download', url, flush=True)
            urllib.request.urlretrieve(url, path)
        raw = gzip.decompress(path.read_bytes())
        offset = 16 if 'images' in name else 8
        array = np.frombuffer(raw[offset:], np.uint8)
        arrays[name] = array.reshape(-1, 784) if 'images' in name else array.reshape(-1, 1)
        files[path.name] = {'url': url, 'sha256': sha256(path)}
    x = np.concatenate([arrays['train-images-idx3-ubyte'], arrays['t10k-images-idx3-ubyte']])
    y = np.concatenate([arrays['train-labels-idx1-ubyte'], arrays['t10k-labels-idx1-ubyte']]).astype(np.int64)
    assert x.shape == (70000, 784) and y.shape == (70000, 1)
    order = np.random.default_rng(0).permutation(60000)
    splits = {'train': order[:50000], 'val': order[50000:], 'test': np.arange(60000, 70000)}
    np.save(out / 'images.npy', x)
    np.save(out / 'labels.npy', y)
    return splits, {'source': 'official MNIST IDX; Google CVDF mirror', 'files': files,
                    'official_train': 60000, 'official_test': 10000,
                    'split_method': 'seed=0 permutation of official train; official test untouched'}


def prepare_donkey(out, data_dir):
    from apps.common.donkey_data import load_donkey_index
    rows = load_donkey_index(data_dir, fixed_throttle=0.3, angle_scale=1.0)
    rows.sort(key=lambda r: int(r[0].name.split('_')[0]))
    # Hash decoded, resized RGB pixels, so duplicate encodings cannot cross splits.
    images = np.lib.format.open_memmap(out / 'images.npy', mode='w+', dtype=np.uint8,
                                      shape=(len(rows), 3, 120, 160))
    labels, records, duplicate_groups = [], [], {}
    for i, (rel, angle, throttle) in enumerate(rows):
        image = cv2.imread(str(data_dir / rel))
        if image is None:
            raise ValueError(f'unreadable image: {rel}')
        image = cv2.cvtColor(cv2.resize(image, (160, 120)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        images[i] = image
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        block = int(rel.name.split('_')[0]) // 100
        duplicate_groups.setdefault(digest, []).append(i)
        labels.append((angle, throttle))
        records.append({'index': i, 'path': rel.as_posix(), 'angle': angle, 'throttle': throttle,
                        'block': block, 'pixel_sha256': digest, 'file_sha256': sha256(data_dir / rel)})
    images.flush()
    del images
    y = np.asarray(labels, np.float32)
    np.save(out / 'labels.npy', y)
    blocks = sorted({r['block'] for r in records})
    parent = {b: b for b in blocks}
    def root(b):
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        return b
    for group in duplicate_groups.values():
        for i in group[1:]:
            parent[root(records[i]['block'])] = root(records[group[0]]['block'])
    groups = sorted({root(b) for b in blocks})
    order = np.random.default_rng(0).permutation(groups)
    nv = max(1, round(len(order) * .1))
    group_sets = {'val': set(order[:nv]), 'test': set(order[nv:2*nv]), 'train': set(order[2*nv:])}
    splits = {name: np.array([i for i, r in enumerate(records) if root(r['block']) in ids], np.int64)
              for name, ids in group_sets.items()}
    for name, indices in splits.items():
        for i in indices:
            records[i]['split'] = name
    (out / 'samples.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
    train_mean = y[splits['train'], 0].mean()
    return splits, {'source': str(data_dir.resolve()), 'catalog_sha256': sha256(data_dir / 'catalog_generated.catalog'),
        'split_method': 'stable numeric filename blocks of 100; union blocks sharing identical decoded pixels; seed=0 group split',
        'limitation': 'generated-road; filename blocks are not verified collection sessions; no real-driving generalization claim',
        'duplicate_images': sum(len(v)-1 for v in duplicate_groups.values()), 'groups': len(groups),
        'angle_range': [float(y[:, 0].min()), float(y[:, 0].max())],
        'throttle_values': np.unique(y[:, 1]).tolist(), 'train_angle_mean': float(train_mean),
        'constant_val_angle_mae': float(np.abs(y[splits['val'], 0] - train_mean).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='.codex/distributed-data-v1')
    ap.add_argument('--data-dir', default='mycar/data')
    ap.add_argument('--tasks', nargs='+', choices=('mnist_mlp', 'donkey_cnn'),
                    default=['mnist_mlp', 'donkey_cnn'])
    args = ap.parse_args()
    for task in args.tasks:
        out = Path(args.out).resolve() / task
        out.mkdir(parents=True, exist_ok=True)
        if (out / 'manifest.json').exists():
            print('already prepared', out, flush=True)
            continue
        splits, meta = prepare_mnist(out) if task == 'mnist_mlp' else prepare_donkey(out, Path(args.data_dir))
        for name, indices in splits.items():
            np.save(out / f'{name}_indices.npy', indices)
        hashes = {p.name: sha256(p) for p in out.glob('*.npy')}
        meta.update(seed=0, task=task, array_sha256=hashes,
                    split_fingerprint=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                    **{f'n_{name}': len(indices) for name, indices in splits.items()})
        (out / 'manifest.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
