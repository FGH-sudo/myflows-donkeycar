"""Audit matched performance runs without claiming whole-batch/shard equivalence."""

import argparse
import json
from pathlib import Path

import numpy as np


def audit(root, baseline='baseline', candidate='notified'):
    checks = []
    expected = {p.parent.relative_to(root / baseline) for p in (root / baseline).glob('*/repeat-*/*/results.json')}
    actual = {p.parent.relative_to(root / candidate) for p in (root / candidate).glob('*/repeat-*/*/results.json')}
    def reference(path):
        return path.with_name('ps-grpc-2') if path.name == 'ps-polling-grpc-2' else path
    matched = {reference(p) for p in actual}
    if matched - expected:
        raise ValueError('candidate contains runs without a matching baseline')
    for result_path in sorted((root / candidate).glob('*/repeat-*/*/results.json')):
        relative = result_path.parent.relative_to(root / candidate)
        before, after = root / baseline / reference(relative), result_path.parent
        old = json.loads((before / 'results.json').read_text(encoding='utf-8'))
        new = json.loads(result_path.read_text(encoding='utf-8'))
        old_config = json.loads((before / 'config.json').read_text(encoding='utf-8'))
        new_config = json.loads((after / 'config.json').read_text(encoding='utf-8'))
        old_initial = json.loads((before / 'rank-0/initial.json').read_text(encoding='utf-8'))
        new_initial = json.loads((after / 'rank-0/initial.json').read_text(encoding='utf-8'))
        for cfg in (old_config, new_config):
            for key in ('artifact_dir', 'run_id', 'ps_wait_strategy'):
                cfg.pop(key, None)
        check = {
            'run': str(relative),
            'config_equal': old_config == new_config,
            'data_equal': old['data_meta'] == new['data_meta'],
            'initial_digest_equal': old_initial['digest'] == new_initial['digest'],
            'final_digest_equal': old['final']['digest'] == new['final']['digest'],
            'status_passed': old['status'] == new['status'] == 'passed',
            'rank_state_consistent': old['rank_state_consistent'] and new['rank_state_consistent'],
            'sample_count_equal': old['epochs'][0]['samples'] == new['epochs'][0]['samples'],
            'clean_exit': not old['alive_pids'] and not new['alive_pids'],
        }
        exact, maximum = True, 0.
        with np.load(before / 'rank-0/final.npz', allow_pickle=False) as a, \
                np.load(after / 'rank-0/final.npz', allow_pickle=False) as b:
            check['state_keys_equal'] = set(a.files) == set(b.files)
            check['metadata_equal'] = json.loads(str(a['metadata'])) == json.loads(str(b['metadata']))
            for key in set(a.files) & set(b.files) - {'metadata'}:
                if a[key].shape != b[key].shape or a[key].dtype != b[key].dtype:
                    exact = False
                    maximum = float('inf')
                    continue
                exact &= np.array_equal(a[key], b[key])
                maximum = max(maximum, float(np.max(np.abs(a[key] - b[key]), initial=0)))
        check['state_arrays_exact'] = bool(exact)
        check['max_state_abs_diff'] = maximum
        losses_equal = True
        for old_path in sorted(before.glob('rank-*/steps.jsonl')):
            new_path = after / old_path.relative_to(before)
            losses = lambda p: [json.loads(line)['loss'] for line in p.read_text(encoding='utf-8').splitlines()]
            losses_equal &= losses(old_path) == losses(new_path)
        check['per_rank_losses_exact'] = bool(losses_equal)
        check['passed'] = all(v for v in check.values() if isinstance(v, bool))
        checks.append(check)
    result = {'baseline': baseline, 'candidate': candidate, 'pairs': len(checks),
              'baseline_runs': len(expected), 'missing': sorted(map(str, expected - matched)),
              'passed': bool(checks) and expected == matched and all(c['passed'] for c in checks), 'checks': checks}
    (root / f'{candidate}_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline', default='baseline')
    parser.add_argument('--candidate', default='notified')
    args = parser.parse_args()
    result = audit(args.out.resolve(), args.baseline, args.candidate)
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
