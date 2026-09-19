"""Paired monitor-on/off full-epoch checks, excluded from architecture tables."""

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.distributed_suite import task_config, execute, resolve_run
from MyFlows.distributed.measurement import write_json


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='docs/experiments/semester_2026_fall/distributed_gpu/20260917')
    ap.add_argument('--tasks',nargs='+',choices=('mnist_mlp','donkey_cnn','donkey_resnet18'),default=['mnist_mlp','donkey_resnet18'])
    args=ap.parse_args()
    root=Path(args.root).resolve()
    rows=[]
    for task in args.tasks:
        for cid,mode,n in [('single-1','single',1),('ring-grpc-4','ring',4)]:
            timings={True:[],False:[]}
            for repeat in range(3):
                for enabled in ((True,False) if repeat%2==0 else (False,True)):
                    config=task_config(root,task)
                    config.update(mode=mode,transport='none' if mode=='single' else 'grpc_proto',train_workers=n,
                        epochs=1,evaluate=False,final_test=False,monitor=enabled,
                        init_checkpoint=str(resolve_run(root,f'{task}/convergence/single-1')/'initial.npz'))
                    _,result=execute(root,f'monitor_overhead/{task}/{cid}/{repeat+1}-{"on" if enabled else "off"}',config)
                    timings[enabled].append(result['epochs'][0]['epoch_train_wall_s'])
            on,off=float(np.median(timings[True])),float(np.median(timings[False]))
            rows.append({'task':task,'config':cid,'on_s':on,'off_s':off,'relative_overhead':on/off-1,
                         'on_repeats':timings[True],'off_repeats':timings[False]})
            write_json(root/'monitor_overhead.json',rows)


if __name__=='__main__':main()
