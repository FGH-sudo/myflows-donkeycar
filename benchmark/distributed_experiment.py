"""Run and archive one measured PS/Ring experiment in a fresh process."""

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import psutil

from MyFlows.distributed.launcher import run_training
from MyFlows.distributed.measurement import write_json


class Resources:
    def __init__(self, out, enabled=True):
        self.out, self.enabled = out, enabled
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def __enter__(self):
        if self.enabled:
            self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        if self.enabled:
            self.thread.join(timeout=6)

    def loop(self):
        me = psutil.Process()
        with (self.out / 'resources.jsonl').open('w', encoding='utf-8', buffering=1) as f:
            while not self.stop.is_set():
                row = {'monotonic_s': time.perf_counter(), 'cpu_utilization_pct': psutil.cpu_percent(),
                       'ram_used_bytes': psutil.virtual_memory().used}
                try:
                    row['process_rss_bytes'] = {str(p.pid): p.memory_info().rss for p in [me]+me.children(recursive=True)
                                                if p.is_running()}
                    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw',
                                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=4,
                                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    fields = result.stdout.strip().splitlines()[0].split(',')
                    row.update(zip(('gpu_utilization_pct','gpu_memory_mib','gpu_temperature_c','gpu_power_w'),
                                   [float(v.strip()) for v in fields]))
                except Exception as exc:
                    row['sampling_error'] = str(exc)
                f.write(json.dumps(row)+'\n')
                self.stop.wait(1.)


def git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True, encoding='utf-8', errors='replace').stdout.strip()


def manifest(config):
    files = list(Path('MyFlows/distributed').rglob('*.py')) + list(Path('benchmark').glob('*distributed*.py'))
    for folder in ('MyFlows/layers', 'MyFlows/core', 'MyFlows/train', 'MyFlows/ops', 'generated/grpc'):
        files.extend(Path(folder).rglob('*.py'))
    files.extend(Path('MyFlows/ops/cuda_native').rglob('*.cpp'))
    files.extend(Path('MyFlows/ops/cuda_native').rglob('*.cu'))
    return {'config': config, 'platform': platform.platform(), 'python': sys.version,
            'bn_statistics_sha256': hashlib.sha256(Path(config['bn_statistics']).read_bytes()).hexdigest() if config.get('bn_statistics') else None,
            'packages': {p: version(p) for p in ('numpy','cupy-cuda12x','torch','grpcio','protobuf','psutil')},
            'root_head': git('rev-parse','HEAD'), 'root_status': git('status','--short'),
            'myflows_head': git('-C','MyFlows','rev-parse','HEAD'), 'myflows_status': git('-C','MyFlows','status','--short'),
            'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'measurement': {'epoch': 'all ranks ready after preparation, through last synchronized update; evaluation excluded',
                'stage_aggregation': 'maximum of per-rank epoch sums; overlapping stages need not sum to epoch wall time',
                'communication': 'gradient synchronization host wall time including encoding, waiting, RPC and aggregation; device copies and optimizer excluded',
                'bytes': 'actual JSON/protobuf message body lengths; socket length prefix separate; HTTP/2/TCP framing unmeasured',
                'hardware': 'single host, all workers share one GPU', 'monitor_interval_s': 1.,
                'readiness_barrier': 'filesystem metadata only, outside training; gradients always use configured network transport'}}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def summarize(out, config, result):
    rank_epochs = [read_jsonl(p) for p in sorted(out.glob('rank-*/epochs.jsonl'))]
    rows = []
    for epoch in range(config['epochs'] if config['task'] != 'synthetic' else 1):
        items = [item for group in rank_epochs for item in group if item['epoch'] == epoch]
        if len(items) != config['train_workers']:
            continue
        total = max(item['end_s'] for item in items)-min(item['start_s'] for item in items)
        row = {'epoch': epoch, 'epoch_train_wall_s': total, 'samples': sum(i['samples'] for i in items),
               'compute_wall_s': max(i['forward_wall_s']+i['backward_wall_s'] for i in items),
               'gradient_sync_s': max(i['gradient_sync_s'] for i in items),
               'samples_per_s': sum(i['samples'] for i in items)/total}
        for key in ('forward_gpu_s','backward_gpu_s','optimizer_gpu_s','optimizer_wall_s','data_s',
                    'input_h2d_wall_s','gradient_d2h_s','gradient_to_device_wall_s','update_confirm_s','digest_s','sync_update_s'):
            values = [i[key] for i in items if i.get(key) is not None]
            row[key] = max(values) if values else None
        for key in ('request_bytes','response_bytes','outgoing_gradient_bytes','socket_framing_bytes',
                    'push_request_bytes','pull_response_bytes','SCATTER_request_bytes','GATHER_request_bytes'):
            row[key] = sum(i.get(key, 0) for i in items)
        val = next((i['val'] for i in items if 'val' in i), {})
        row.update({'val_'+k:v for k,v in val.items() if k not in ('split','evaluation_s')})
        steps = [r for p in out.glob('rank-*/steps.jsonl') for r in read_jsonl(p) if r['epoch']==epoch]
        row['rank_step_median_s'] = float(np.median([r['step_wall_s'] for r in steps]))
        row['rank_step_p95_s'] = float(np.percentile([r['step_wall_s'] for r in steps],95))
        rows.append(row)
    if rows:
        with (out/'epochs.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    finals = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(out.glob('rank-*/final.json'))]
    if result.get('status') == 'passed' and (len(finals)!=config['train_workers'] or len({f['digest'] for f in finals})!=1):
        result.update(status='failed', error='missing or inconsistent final rank state')
    return {'status': result.get('status'), 'error':result.get('error'), 'epochs':rows,
            'elapsed_s':result.get('elapsed_s'),'cleanup_s':result.get('cleanup_s'), 'final':finals[0] if finals else {},
            'rank_state_consistent':bool(finals) and len({f['digest'] for f in finals})==1,
            'data_meta':result.get('data_meta'), 'alive_pids':result.get('alive_pids'), 'exitcodes':result.get('exitcodes')}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    args=ap.parse_args()
    out=Path(args.out).resolve()
    config=json.loads(Path(args.config).read_text(encoding='utf-8'))
    for parent in out.parents:
        scope_file=parent/'execution_scope.json'
        if scope_file.exists():
            scope=json.loads(scope_file.read_text(encoding='utf-8'))
            if config['task'] not in scope['tasks']:
                print(f"CANCELLED: task {config['task']} excluded by user scope in {scope_file}",flush=True)
                raise SystemExit(75)
            break
    out.mkdir(parents=True, exist_ok=False)
    config.update(artifact_dir=str(out), run_id=out.name)
    config.setdefault('train_workers',1)
    config.setdefault('profile',True)
    config.setdefault('collect_history',False)
    write_json(out/'config.json',config)
    write_json(out/'manifest.json',manifest(config))
    if config.get('prepared_dir'):
        write_json(out/'data_manifest.json',json.loads((Path(config['prepared_dir'])/'manifest.json').read_text(encoding='utf-8')))
    result={}
    try:
        with Resources(out, config.get('monitor',True)):
            result=run_training(**config)
        with (out/'events.jsonl').open('w',encoding='utf-8') as f:
            for event in result.get('events',[]):
                if event.get('phase')!='history':
                    f.write(json.dumps(event,default=lambda v:v.tolist())+'\n')
        summary=summarize(out,config,result)
        write_json(out/'results.json',summary)
        if summary['status']!='passed':
            raise RuntimeError(summary.get('error'))
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    except BaseException as exc:
        write_json(out/'failure.json',{'error':str(exc),'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':
    main()
