"""Repeat the frozen single/PS-2/Ring-2 workloads serially in fresh processes."""

import argparse
import json
import hashlib
import os
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'docs/experiments/semester_2026_fall/distributed_gpu/20260919_optimized'


def summarize(out):
    groups = {}
    for path in sorted(out.glob('*/*/repeat-*/*/results.json')):
        result = json.loads(path.read_text(encoding='utf-8'))
        config = json.loads(path.with_name('config.json').read_text(encoding='utf-8'))
        phase, task, repeat, mode = path.relative_to(out).parts[:4]
        if result['status'] != 'passed' or not result['rank_state_consistent']:
            raise RuntimeError(f'failed or inconsistent run: {path}')
        if len(result['epochs']) != 1:
            raise RuntimeError(f'expected one complete measured epoch: {path}')
        row = result['epochs'][0]
        groups.setdefault((phase, task, mode), []).append({
            'run': str(path.parent.relative_to(out)), 'repeat': repeat,
            'profile': config['profile'], **row,
        })
    summary = []
    for (phase, task, mode), rows in sorted(groups.items()):
        fields = [k for k, v in rows[0].items() if k.endswith('_s') and isinstance(v, (int, float))]
        summary.append({
            'phase': phase, 'task': task, 'mode': mode, 'repeats': len(rows),
            'median': {k: statistics.median(r[k] for r in rows) for k in fields},
            'min_epoch_s': min(r['epoch_train_wall_s'] for r in rows),
            'max_epoch_s': max(r['epoch_train_wall_s'] for r in rows),
            'samples': sorted({r['samples'] for r in rows}), 'runs': rows,
        })
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phase', required=True)
    parser.add_argument('--tasks', nargs='+', choices=['mnist_mlp', 'donkey_resnet18'],
                        default=['mnist_mlp', 'donkey_resnet18'])
    parser.add_argument('--modes', nargs='+', choices=['single', 'ps', 'ring'], default=['single', 'ps', 'ring'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--include-polling-ablation', action='store_true',
                        help='Interleave a benchmark-only reproduction of the old PS wait loop.')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    out = args.out.resolve()
    if args.include_polling_ablation:
        args.modes.append('ps-polling')
    for repeat in range(args.repeats):
        modes = args.modes[repeat % len(args.modes):] + args.modes[:repeat % len(args.modes)]
        for task in args.tasks:
            for mode in modes:
                label = 'single-1' if mode == 'single' else f'{mode}-grpc-2'
                template = 'ps-grpc-2' if mode == 'ps-polling' else label
                source = FROZEN / task / 'performance/repeat-1' / f'{template}.config.json'
                config = json.loads(source.read_text(encoding='utf-8'))
                config['profile'] = True
                if mode in ('ps', 'ps-polling'):
                    config['ps_wait_strategy'] = 'notify' if mode == 'ps' else 'poll'
                run = out / args.phase / task / f'repeat-{repeat + 1}' / label
                run.parent.mkdir(parents=True, exist_ok=True)
                if run.exists():
                    raise FileExistsError(f'refusing to overwrite evidence: {run}')
                config_path = run.with_suffix('.config.json')
                config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
                print(f'START {run.relative_to(out)}', flush=True)
                env = os.environ.copy()
                env.pop('MYFLOWS_BENCHMARK_PS_WAIT', None)
                module = 'benchmark.distributed_experiment'
                if mode == 'ps-polling':
                    env['MYFLOWS_BENCHMARK_PS_WAIT'] = 'poll20ms'
                    module = 'benchmark.ps_wait_ablation'
                with run.with_suffix('.log').open('w', encoding='utf-8') as log:
                    subprocess.run([sys.executable, '-m', module,
                                    '--config', str(config_path), '--out', str(run)], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200, env=env)
                (run / 'benchmark_variant.json').write_text(json.dumps({
                    'module': module, 'wait': 'poll20ms' if mode == 'ps-polling' else 'production',
                    'ablation_source_sha256': hashlib.sha256(
                        (ROOT / 'benchmark/ps_wait_ablation.py').read_bytes()).hexdigest(),
                }, indent=2), encoding='utf-8')
                result = json.loads((run / 'results.json').read_text(encoding='utf-8'))
                print(f"DONE {label}: {result['epochs'][0]['epoch_train_wall_s']:.6f} s", flush=True)
                summarize(out)


if __name__ == '__main__':
    main()
