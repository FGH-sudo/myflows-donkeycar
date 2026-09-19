"""Validate retained MNIST runs and the controls needed for fair comparison."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark.distributed_compare import MATRIX
from benchmark.distributed_experiment import read_jsonl
from benchmark.distributed_suite import task_config, resolve_run
from MyFlows.distributed.measurement import write_json


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='docs/experiments/semester_2026_fall/distributed_gpu/20260917')
    ap.add_argument('--task', choices=('mnist_mlp', 'donkey_resnet18'), default='mnist_mlp')
    args = ap.parse_args()
    root = Path(args.root).resolve()
    task = args.task
    base = resolve_run(root, f'{task}/convergence/single-1')
    common = task_config(root, task)
    data = load(base/'data_manifest.json')
    manifest = load(base/'manifest.json')
    measured_manifest = load(resolve_run(root, f'{task}/performance/repeat-1/single-1')/'manifest.json')
    code = {k: v for k, v in measured_manifest['source_sha256'].items() if k.replace('\\', '/').startswith(('MyFlows/', 'generated/grpc/'))}
    assert code
    with np.load(base/'initial.npz', allow_pickle=False) as saved:
        initial = {key: saved[key].copy() for key in saved.files}
    prepared = Path(load(base/'config.json')['prepared_dir'])
    hashes = {name: hashlib.sha256((prepared/name).read_bytes()).hexdigest() for name in data['array_sha256']}
    assert hashes == data['array_sha256'], 'cached MNIST arrays changed'
    indices = {s: np.load(prepared/f'{s}_indices.npy') for s in ('train', 'val', 'test')}
    assert len(np.intersect1d(indices['train'], indices['val'])) == 0
    # Official test images reside in the same prepared array after the 60k train part.
    assert len(np.intersect1d(indices['train'], indices['test'])) == 0
    assert len(np.intersect1d(indices['val'], indices['test'])) == 0
    checks = []
    source_changes = []
    batch = common['global_batch']
    train_samples = data['n_train']
    steps_per_epoch = train_samples//batch
    for phase in ('convergence', 'performance/repeat-1', 'performance/repeat-2', 'performance/repeat-3'):
        epochs = common['epochs'] if phase == 'convergence' else 1
        for cid, mode, transport, n in MATRIX:
            directory = resolve_run(root, f'{task}/{phase}/{cid}')
            cfg, result = load(directory/'config.json'), load(directory/'results.json')
            assert (cfg['mode'], cfg['transport'], cfg['train_workers']) == (mode, transport, n)
            for key, value in dict(task=task, device='cuda', optimizer='adam', seed=0,
                                   learning_rate=common['learning_rate'], global_batch=batch, epochs=epochs,
                                   profile=True, monitor=True, steps=100000).items():
                assert cfg[key] == value, (directory, key)
            assert load(directory/'data_manifest.json') == data, directory
            current = load(directory/'manifest.json')
            if phase != 'convergence':
                assert {k: current['source_sha256'].get(k) for k in code} == code, directory
            else:
                changed = [k for k in code if k in current['source_sha256'] and current['source_sha256'][k] != code[k]]
                if changed:
                    source_changes.append({'run': str(directory.relative_to(root)), 'files': changed})
            if task == 'donkey_resnet18':
                assert current['bn_statistics_sha256'] == measured_manifest['bn_statistics_sha256']
            assert current['packages'] == manifest['packages'], directory
            with np.load(directory/'initial.npz', allow_pickle=False) as actual:
                assert set(actual.files) == set(initial), directory
                for key, value in initial.items():
                    np.testing.assert_array_equal(actual[key], value, err_msg=f'{directory}: {key}')
            assert result['status'] == 'passed' and not result['alive_pids'], directory
            assert result['rank_state_consistent'] and len(result['epochs']) == epochs, directory
            finals = []
            for rank in range(n):
                rank_dir = directory/f'rank-{rank}'
                start = load(rank_dir/'initial.json')
                end = load(rank_dir/'final.json')
                assert start['device'] == end['device'] == 'cuda', rank_dir
                assert start['digest'] == load(base/'rank-0/initial.json')['digest'], rank_dir
                assert all(v['module'].startswith('cupy') and v['dtype'] == 'float32'
                           for v in start['parameters'].values()), rank_dir
                assert end['version'] == end['optimizer']['t'] == epochs*steps_per_epoch, rank_dir
                assert all(v['module'].startswith('cupy') and v['dtype'] == 'float32'
                           for values in end['optimizer_arrays'].values() for v in values.values()), rank_dir
                steps = read_jsonl(rank_dir/'steps.jsonl')
                assert [r['step'] for r in steps] == list(range(epochs*steps_per_epoch)), rank_dir
                per_epoch = read_jsonl(rank_dir/'epochs.jsonl')
                assert len(per_epoch) == epochs and all(r['steps'] == steps_per_epoch and r['samples'] == steps_per_epoch*batch//n for r in per_epoch)
                assert (rank_dir/'final.npz').is_file(), rank_dir
                finals.append(end['digest'])
                if transport == 'grpc_proto':
                    assert end['traffic'].get('socket_framing_bytes', 0) == 0, rank_dir
                    if mode == 'ring':
                        assert end['traffic']['confirm_update_request_bytes'] > 0, rank_dir
            assert len(set(finals)) == 1, directory
            for ep in result['epochs']:
                assert ep['samples'] == steps_per_epoch*batch and ep['epoch_train_wall_s'] > 0, directory
            if phase == 'convergence':
                final = result['final']
                assert final['val']['samples'] == data['n_val'] and final['test']['samples'] == data['n_test']
                assert final['train']['samples'] == train_samples
                if task == 'mnist_mlp':
                    assert final['val']['accuracy'] >= .95
                else:
                    assert set(final['backends']) == {'cuda_native_cublas'}
                    with np.load(directory/'rank-0/final.npz') as saved:
                        for name in initial:
                            if name.startswith('buffer::'):
                                np.testing.assert_array_equal(saved[name], initial[name])
            else:
                assert not cfg['evaluate'] and not cfg['final_test']
                assert 'test' not in result['final'] and 'val' not in result['final']
            checks.append({'run': str(directory.relative_to(root)), 'epochs': epochs,
                           'global_steps': epochs*steps_per_epoch, 'workers': n, 'passed': True})
    output = {'passed': True, 'runs': checks, 'run_count': len(checks),
              'task': task, 'split_fingerprint': data['split_fingerprint'],
              'array_hashes_verified': hashes, 'same_initial_state': True,
              'performance_same_runtime_source': True, 'same_dependencies': True,
              'convergence_source_changes_relative_to_performance': source_changes,
              'runtime_source_sha256': code,
              'source_scope': 'Performance runs compare all recorded MyFlows and generated/grpc sources; earlier convergence includes additive ResNet registration/BN checkpoint changes, listed explicitly'}
    write_json(root/f'{task}_artifact_audit.json', output)
    print(f'{task} artifact audit: {len(checks)}/28 runs passed; data, initialization, steps and GPU state matched; performance sources matched')


if __name__ == '__main__':
    main()
