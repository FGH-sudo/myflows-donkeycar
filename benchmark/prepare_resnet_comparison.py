"""Calibrate existing ResNet BN using training images, then freeze shared moments."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from benchmark.distributed_suite import base_config
from MyFlows.core.device import asnumpy, xp
from MyFlows.distributed.measurement import write_json
from MyFlows.distributed.session import TrainSession


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='docs/experiments/semester_2026_fall/distributed_gpu/20260917')
    ap.add_argument('--width',type=int,default=16)
    ap.add_argument('--batch',type=int,default=128)
    args=ap.parse_args();root=Path(args.root).resolve()
    directory=root/'resnet_preflight'/f'width-{args.width}-batch-{args.batch}'
    directory.mkdir(parents=True,exist_ok=False)
    config=base_config('donkey_resnet18')
    config.update(base_width=args.width,global_batch=args.batch)
    session=TrainSession(config)
    data=session.task.make_data(config)
    model=session.task.model
    model.train(True)
    start=time.perf_counter()
    for index in range(16):
        first=index*32
        session.x.value=xp.asarray(data['train'][0][first:first+32])
        session.y.value=xp.asarray(data['train'][1][first:first+32])
        session.graph.forward()
    xp.cuda.Stream.null.synchronize()
    model.eval()
    bn_file=directory/'frozen_bn.npz'
    np.savez(bn_file,**{k:asnumpy(v) for k,v in session.buffers.items()})
    config['bn_statistics']=str(bn_file)
    rows=[]
    for index in range(3):
        x,y=data['train'][0][:args.batch],data['train'][1][:args.batch]
        row=session.forward_backward(x,y,materialize_cpu_gradients=False)
        session.apply_global_gradients(row['device_gradients'],f'preflight-{index}')
        rows.append({'loss':row['loss'],'timings':row['timings'], 'update':session.update_timings})
    info={'model':session.task.model_metadata, 'config':config, 'calibration_training_samples':512,
          'validation_or_test_used':False, 'data_split_fingerprint':data['meta']['split_fingerprint'],
          'elapsed_s':time.perf_counter()-start,'steps':rows,
          'allocator_reserved_mib':xp.get_default_memory_pool().total_bytes()/2**20,
          'cuda_free_and_total_bytes':list(xp.cuda.runtime.memGetInfo())}
    write_json(directory/'results.json',info)
    write_json(directory/'candidate_config.json',config)
    print(json.dumps(info,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
