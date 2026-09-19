"""Compare full/sharded GPU gradients at exactly the same saved ResNet state."""

import json
from pathlib import Path

import numpy as np

from benchmark.distributed_suite import task_config
from MyFlows.distributed.measurement import load_state, write_json
from MyFlows.distributed.session import TrainSession
from MyFlows.distributed.schedule import BatchCursor
from MyFlows.distributed.protocol import aggregate_weighted
from MyFlows.core.device import asnumpy


def delta(left, right):
    maximum=max((float(np.max(np.abs(left[k]-right[k]))),k) for k in left)
    bad=sum(int(np.sum(np.abs(left[k]-right[k]) > 1e-4+1e-3*np.abs(right[k]))) for k in left)
    return {'max_absolute_error':maximum[0],'parameter':maximum[1],'failing_elements':bad}


def branches(session):
    result={}
    for i,node in enumerate(session.graph.nodes):
        if type(node).__name__=='ReLU':
            result[f'{i}:ReLU']=asnumpy(node.parents[0].value>0,copy=True)
        if type(node).__name__=='MaxPool2d_Op':
            context=node._native_context
            per_image=int(np.prod(context.input_shape[1:]))
            result[f'{i}:MaxPool']=asnumpy(context.argmax,copy=True)%per_image
    return result


def main():
    root=Path('docs/experiments/semester_2026_fall/distributed_gpu/20260917').resolve()
    cfg=task_config(root,'donkey_resnet18')
    cfg.update(optimizer='mbgd',global_batch=32,epochs=1,steps=20,evaluate=False,final_test=False,monitor=False)
    session=TrainSession(cfg)
    data=session.task.make_data(cfg);cursor=BatchCursor(session.task,data,cfg)
    base=root/'numeric/donkey_resnet18/mbgd'
    output=[]
    for step in (0,3,10,19):
        states=[('single',base/'single-1'/('initial.npz' if step==0 else f'rank-0/step-{step-1:04d}.npz')),
                ('distributed',base/'ps-socket_json-2'/('initial.npz' if step==0 else f'rank-0/step-{step-1:04d}.npz'))]
        full_grads={};full_branches={}
        for label,path in states:
            load_state(session,path)
            x,y=cursor.get(0,step)
            full=session.forward_backward(x,y)['gradients'];full_grads[label]=full
            full_branches[label]=branches(session)
            parts=[];part_branches=[]
            for sl in (slice(0,20),slice(20,32)):
                part=session.forward_backward(x[sl],y[sl])
                parts.append({'n_samples':len(x[sl]),'gradients':part['gradients'],'loss':part['loss']})
                part_branches.append(branches(session))
            average,_,_=aggregate_weighted(parts)
            changed={key:int(np.sum(value!=np.concatenate([p[key] for p in part_branches],axis=0)))
                     for key,value in full_branches[label].items()}
            row={'step':step,'state':label,'same_state_full_vs_shards':delta(average,full),
                 'changed_branches':{k:v for k,v in changed.items() if v}}
            output.append(row);print(row,flush=True)
        changed={key:int(np.sum(value!=full_branches['distributed'][key])) for key,value in full_branches['single'].items()}
        row={'step':step,'cross_state_full_batch':delta(full_grads['single'],full_grads['distributed']),
             'changed_branches':{k:v for k,v in changed.items() if v}}
        output.append(row);print(row,flush=True)
    write_json(root/'resnet_gradient_diagnosis.json',output)


if __name__=='__main__':main()
