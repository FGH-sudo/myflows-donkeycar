"""Serial continuation: ResNet preflight, retained checks, both course tables."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import psutil

from benchmark.distributed_suite import execute, numerical, state_checks, faults, train, performance
from MyFlows.distributed.measurement import write_json


def run(root, module, *args):
    command=[sys.executable,'-X','utf8','-m',module,*args]
    print('STAGE',module,*args,flush=True)
    log=root/'suite_logs'/(module.rsplit('.',1)[-1]+'-'+str(len(list((root/'suite_logs').glob('*.log'))))+'.log')
    with log.open('w',encoding='utf-8') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{module} failed; {log}')
    print('STAGE PASS',module,flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='docs/experiments/semester_2026_fall/distributed_gpu/20260917')
    ap.add_argument('--phase',choices=('prepare','validate','train','performance','extras','report','all'),default='all')
    ap.add_argument('--wait-pid',type=int,help='Wait for an existing serial GPU suite before starting')
    args=ap.parse_args();root=Path(args.root).resolve();(root/'suite_logs').mkdir(parents=True,exist_ok=True)
    if args.wait_pid:
        print('WAIT FOR EXISTING SUITE',args.wait_pid,flush=True)
        try:
            previous=psutil.Process(args.wait_pid)
            while previous.is_running():
                time.sleep(2)
        except psutil.NoSuchProcess:
            pass
    write_json(root/'execution_scope.json',{'tasks':['synthetic','mnist_mlp','donkey_cnn','donkey_resnet18'],
               'source':'User requested resumed MNIST plus existing ResNet18 road comparison; topology uses gRPC/protobuf'})
    if args.phase in ('prepare','all'):
        candidate=root/'resnet_preflight/width-16-batch-128/candidate_config.json'
        if not candidate.exists():
            run(root,'benchmark.prepare_resnet_comparison','--root',str(root))
        config=json.loads(candidate.read_text(encoding='utf-8'))
        out,pilot=execute(root,'resnet_preflight/pilot-5epochs',dict(config,mode='single',transport='none',train_workers=1,final_test=False))
        initial=json.loads((out/'rank-0/initial.json').read_text(encoding='utf-8'))
        data=json.loads((out/'data_manifest.json').read_text(encoding='utf-8'))
        final=pilot['final']
        if not (final['train']['loss']<=initial['train']['loss']*.7 and final['val']['angle_mae']<=data['constant_val_angle_mae']*.9):
            raise AssertionError(f'ResNet pilot quality not met; preserve {out} and diagnose before freezing')
        frozen=root/'frozen_task_configs.json'
        chosen=json.loads(frozen.read_text(encoding='utf-8')) if frozen.exists() else {}
        chosen['donkey_resnet18']=config
        write_json(frozen,chosen)
        write_json(root/'resnet_preflight/freeze_decision.json',{'pilot':str(out.relative_to(root)),
                   'train_loss':final['train']['loss'],'initial_train_loss':initial['train']['loss'],
                   'val_angle_mae':final['val']['angle_mae'],'constant_val_angle_mae':data['constant_val_angle_mae'],
                   'test_set_used':False,'config':config})
        print('FROZEN RESNET',final['val'],flush=True)
    if args.phase in ('validate','all'):
        run(root,'tools.run_tests','--scope','all')
        numerical(root,('donkey_resnet18',))
        state_checks(root,('donkey_resnet18',))
        faults(root)
        run(root,'benchmark.validate_ring_volume','--root',str(root))
    if args.phase in ('train','all'):
        train(root,('mnist_mlp','donkey_resnet18'))
    if args.phase in ('performance','all'):
        performance(root,('mnist_mlp','donkey_resnet18'))
    if args.phase in ('extras','all'):
        for tool in ('nsys','ncu'):
            run(root,'benchmark.profile_distributed','--root',str(root),'--task','donkey_resnet18','--tool',tool)
        run(root,'benchmark.distributed_monitor_overhead','--root',str(root),'--tasks','mnist_mlp','donkey_resnet18')
    if args.phase in ('report','all'):
        for task in ('mnist_mlp','donkey_resnet18'):
            run(root,'benchmark.audit_distributed_artifacts','--root',str(root),'--task',task)
        run(root,'benchmark.report_distributed','--root',str(root))


if __name__=='__main__':
    # Durable status remains readable if the desktop turn is interrupted.
    status_root=Path('docs/experiments/semester_2026_fall/distributed_gpu/20260917')
    if '--root' in sys.argv:
        status_root=Path(sys.argv[sys.argv.index('--root')+1])
    started=time.time()
    previous_status=status_root/'continuation_status.json'
    if previous_status.exists():
        old=json.loads(previous_status.read_text(encoding='utf-8'))
        write_json(status_root/f'continuation_status_attempt-{int(started)}.json',old)
    write_json(status_root/'continuation_status.json',{'status':'running','started_unix_s':started,'argv':sys.argv})
    try:
        main()
    except BaseException as exc:
        write_json(status_root/'continuation_status.json',{'status':'failed','started_unix_s':started,
                   'finished_unix_s':time.time(),'error':str(exc),'traceback':traceback.format_exc()})
        raise
    else:
        write_json(status_root/'continuation_status.json',{'status':'completed','started_unix_s':started,
                   'finished_unix_s':time.time(),'argv':sys.argv})
