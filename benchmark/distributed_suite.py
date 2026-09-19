"""Reproducible serial GPU numerical, convergence and performance matrix."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from benchmark.distributed_compare import MATRIX
from MyFlows.distributed.measurement import write_json


def resolve_run(root, label):
    index=root/'run_index.json'
    if index.exists():
        target=json.loads(index.read_text(encoding='utf-8')).get(str(label))
        if target:
            return root/target
    return root/label


def index_run(root, label, out):
    path=root/'run_index.json'
    index=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    index[str(label)]=out.relative_to(root).as_posix()
    write_json(path,index)


def execute(root, label, config):
    out = root / label
    original = out
    attempt = 1
    while out.exists():
        result_file = out / 'results.json'
        if result_file.exists():
            result = json.loads(result_file.read_text(encoding='utf-8'))
            if result.get('status') == 'passed':
                saved=json.loads((out/'config.json').read_text(encoding='utf-8'))
                same=all(saved.get(key)==value for key,value in config.items())
                if same and config.get('bn_statistics'):
                    saved_manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
                    same=saved_manifest.get('bn_statistics_sha256')==hashlib.sha256(Path(config['bn_statistics']).read_bytes()).hexdigest()
                if same:
                    index_run(root,label,out)
                    print('reuse',out,flush=True)
                    return out, result
        attempt += 1
        out = original.with_name(original.name + f'-attempt{attempt}')
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = out.with_suffix('.config.json')
    write_json(cfg,config)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONIOENCODING='utf-8')
    command = [sys.executable, '-X','utf8','-m','benchmark.distributed_experiment','--config',str(cfg),'--out',str(out)]
    print('RUN', label, flush=True)
    with out.with_suffix('.log').open('w',encoding='utf-8') as log:
        proc = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
    if proc.returncode:
        raise RuntimeError(f'{out} failed; see {out.with_suffix(".log")}')
    result=json.loads((out/'results.json').read_text(encoding='utf-8'))
    index_run(root,label,out)
    print('PASS',label,'elapsed',round(result['elapsed_s'],2),'val',result['final'].get('val'),flush=True)
    return out,result


def base_config(task):
    config = dict(task=task,device='cuda',optimizer='adam',learning_rate=.001,seed=0,
        global_batch=128 if task=='mnist_mlp' else 32,epochs=10 if task=='mnist_mlp' else 20,
        steps=100000,timeout=180.,profile=True,evaluate=True,final_test=True,
        monitor=True,poll_interval_s=.001,collect_history=False,
        prepared_dir=str(Path('.codex/distributed-data-v1','donkey_cnn' if task=='donkey_resnet18' else task).resolve()))
    if task == 'donkey_resnet18':
        config.update(base_width=16, resnet_stem='imagenet', bn_mode='frozen', global_batch=128,
                      eval_batch=32, epochs=5, learning_rate=.0003, save_epoch_checkpoints=True,
                      timeout=300., heartbeat_timeout_s=60.)
    return config


def task_config(root, task):
    config = base_config(task)
    frozen = root/'frozen_task_configs.json'
    if frozen.exists():
        config.update(json.loads(frozen.read_text(encoding='utf-8')).get(task, {}))
    return config


def merge_checks(path, rows, key='run'):
    previous = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    merged = {row[key]: row for row in previous}
    merged.update({row[key]: row for row in rows})
    write_json(path, list(merged.values()))


def numerical(root, tasks=('mnist_mlp','donkey_cnn')):
    checks=[]
    for task in ('synthetic',*tasks):
        for opt in ('mbgd','adam'):
            base=task_config(root, task)
            base.update(optimizer=opt,global_batch=32,epochs=1,steps=20,evaluate=False,
                        final_test=False,monitor=False,numerical_snapshots=True)
            if task=='synthetic':
                base.pop('prepared_dir')
            if task=='donkey_resnet18':
                base['record_local_gradients']=True
            reference,_=execute(root,f'numeric/{task}/{opt}/single-1',dict(base,mode='single',transport='none',train_workers=1))
            for mode,transport in (('ps','socket_json'),('ps','grpc_proto'),('ring','grpc_proto')):
                for n in (1,2,4):
                    config=dict(base,mode=mode,transport=transport,train_workers=n,
                                shard_sizes={1:[32],2:[20,12],4:[11,9,7,5]}[n])
                    label=f'numeric/{task}/{opt}/{mode}-{transport}-{n}'
                    out,result=execute(root,label,config)
                    largest={}
                    if task=='donkey_resnet18':
                        from benchmark.validate_resnet_steps import check_resnet_steps
                        largest=check_resnet_steps(out,reference)
                    for rank in range(n if task!='donkey_resnet18' else 0):
                        snapshots=sorted((out/f'rank-{rank}').glob('step-*.npz'))
                        if len(snapshots)!=20:
                            raise AssertionError(f'{label} rank {rank}: missing snapshots')
                        for snap in snapshots:
                            with np.load(snap) as actual, np.load(reference/'rank-0'/snap.name) as expect:
                                for key in expect.files:
                                    if key=='metadata':
                                        a,b=json.loads(str(actual[key])),json.loads(str(expect[key]))
                                        if a!=b:
                                            raise AssertionError(f'{label} optimizer/version metadata mismatch: {a} vs {b}')
                                        continue
                                    delta=float(np.max(np.abs(actual[key]-expect[key])))
                                    kind=key.split('::')[0]
                                    largest[kind]=max(largest.get(kind,0.),delta)
                                    np.testing.assert_allclose(actual[key],expect[key],atol=1e-4,rtol=1e-3,
                                                               err_msg=f'{label} rank {rank} {snap.name} {key}')
                    checks.append({'run':label,'workers':n,'steps':20,'shards':config['shard_sizes'],
                                   'artifact_run':out.relative_to(root).as_posix(),
                                   'reference_kind':'recorded_shards_and_cpu_updates' if task=='donkey_resnet18' else 'independent_full_batch_trajectory',
                                   'max_absolute_error':largest,'passed':True})
                    merge_checks(root/'numeric_checks.json',checks)


def train(root, tasks=('mnist_mlp','donkey_cnn')):
    outcomes=[]
    for task in tasks:
        baseline=None
        for cid,mode,transport,n in MATRIX:
            out,result=execute(root,f'{task}/convergence/{cid}',dict(task_config(root, task),mode=mode,transport=transport,train_workers=n))
            final=result['final'];val=final['val']
            if cid=='single-1':baseline=val
            if task=='mnist_mlp':
                passed=val['accuracy']>=.95 and abs(val['accuracy']-baseline['accuracy'])<=.01
            else:
                meta=json.loads((out/'data_manifest.json').read_text(encoding='utf-8'))
                initial=json.loads((out/'rank-0/initial.json').read_text(encoding='utf-8'))
                passed=(final['train']['loss']<=initial['train']['loss']*.7
                        and val['angle_mae']<=meta['constant_val_angle_mae']*.9
                        and abs(val['angle_mae']-baseline['angle_mae'])<=max(.001,baseline['angle_mae']*.05))
            outcomes.append({'run':str(out),'task':task,'config':cid,'quality_passed':passed,'val':val,'test':final.get('test')})
            merge_checks(root/'quality_checks.json',outcomes)
            if not passed:
                raise AssertionError(f'quality gate failed: {out}')


def performance(root, tasks=('mnist_mlp','donkey_cnn')):
    for repeat in range(3):
        matrix=list(MATRIX)
        if repeat==1:matrix.reverse()
        if repeat==2:matrix=matrix[3:]+matrix[:3]
        for task in tasks:
            for cid,mode,transport,n in matrix:
                base=task_config(root, task)
                base.update(epochs=1,evaluate=False,final_test=False,
                    init_checkpoint=str((resolve_run(root,f'{task}/convergence/single-1')/'initial.npz').resolve()))
                execute(root,f'{task}/performance/repeat-{repeat+1}/{cid}',
                        dict(base,mode=mode,transport=transport,train_workers=n))


def state_checks(root, tasks=('mnist_mlp','donkey_cnn')):
    checks=[]
    for task in tasks:
        base=task_config(root, task)
        base.update(global_batch=32,epochs=2,steps=2,evaluate=False,final_test=False,
                    monitor=False,numerical_snapshots=True)
        if task=='donkey_resnet18':
            base['record_local_gradients']=True
        reference,_=execute(root,f'state/{task}/uninterrupted',dict(base,mode='single',transport='none',train_workers=1))
        base.update(epochs=1,data_epoch=1,init_checkpoint=str(reference/'rank-0/step-0001.npz'))
        for mode,transport in (('ps','socket_json'),('ps','grpc_proto'),('ring','grpc_proto')):
            for n in (1,2,4):
                label=f'state/{task}/{mode}-{transport}-{n}'
                out,_=execute(root,label,dict(base,mode=mode,transport=transport,train_workers=n))
                worst=0.
                if task=='donkey_resnet18':
                    from benchmark.validate_resnet_steps import check_resnet_steps
                    measured=check_resnet_steps(out,reference,steps=2,start_step=2)
                    worst=max(measured.values())
                for rank in range(n if task!='donkey_resnet18' else 0):
                    for step in (2,3):
                        with np.load(reference/f'rank-0/step-{step:04d}.npz') as ref, np.load(out/f'rank-{rank}/step-{step:04d}.npz') as got:
                            for key in ref.files:
                                if key=='metadata':
                                    assert str(ref[key])==str(got[key]),label
                                else:
                                    np.testing.assert_allclose(got[key],ref[key],atol=1e-4,rtol=1e-3,err_msg=label+' '+key)
                                    worst=max(worst,float(np.max(np.abs(got[key]-ref[key]))))
                checks.append({'run':label,'restored_version':2,'final_version':4,'passed':True,'max_absolute_error':worst})
                merge_checks(root/'state_checks.json',checks)


def faults(root):
    from MyFlows.distributed.launcher import run_training
    import time
    if (root/'fault_checks.json').exists():
        write_json(root/f'fault_checks_attempt-{time.time_ns()}.json',
                   json.loads((root/'fault_checks.json').read_text(encoding='utf-8')))
    results=[]
    for mode,transport in (('ps','socket_json'),('ps','grpc_proto'),('ring','grpc_proto')):
        names=['crash','timeout','schema','duplicate']+(['heartbeat_loss','retry'] if mode=='ps' else ['gather_crash'])
        for fault in names:
            result=run_training(mode=mode,transport=transport,train_workers=2,task='synthetic',
                device='cpu',steps=4,global_batch=32,timeout=4,heartbeat_timeout_s=2,
                heartbeat_interval_s=.5,fault=fault,collect_history=False)
            expected='passed' if fault=='retry' or (mode=='ring' and fault=='duplicate') else 'failed'
            passed=result['status']==expected and not result['alive_pids'] and result['cleanup_s']<5
            row={k:result[k] for k in ('status','error','alive_pids','cleanup_s','events') if k in result}
            row.update(mode=mode,transport=transport,fault=fault,expected=expected,passed=passed)
            results.append(row);write_json(root/'fault_checks.json',results)
            print('FAULT',mode,transport,fault,passed,flush=True)
            if not passed:raise AssertionError(row)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='docs/experiments/semester_2026_fall/distributed_gpu/20260917')
    ap.add_argument('--phase',choices=('numeric','state','train','performance','fault','all'),default='all')
    ap.add_argument('--tasks',nargs='+',choices=('mnist_mlp','donkey_cnn','donkey_resnet18'))
    args=ap.parse_args();root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    scope_file=root/'execution_scope.json'
    if args.tasks:
        tasks=args.tasks
        write_json(scope_file,{'tasks':['synthetic',*tasks], 'source':'explicit suite --tasks selection'})
    elif scope_file.exists():
        tasks=[t for t in json.loads(scope_file.read_text(encoding='utf-8'))['tasks'] if t!='synthetic']
    else:
        tasks=['mnist_mlp','donkey_cnn']
    for phase,fn in [('numeric',numerical),('state',state_checks),('fault',faults),('train',train),('performance',performance)]:
        if args.phase in (phase,'all'):
            fn(root) if phase=='fault' else fn(root,tasks)


if __name__=='__main__':
    main()
