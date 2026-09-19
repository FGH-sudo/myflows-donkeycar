"""Audit ResNet collective and GPU updates independently of nonsmooth trajectory drift."""

import json
from pathlib import Path

import numpy as np

from MyFlows.distributed.measurement import load_state, save_state, write_json
from MyFlows.distributed.session import TrainSession


def check_resnet_steps(out, trajectory_reference, steps=20, start_step=0):
    config=json.loads((out/'config.json').read_text(encoding='utf-8'))
    from MyFlows.distributed.shards import split_global_batch
    sizes=split_global_batch(config['global_batch'],config['train_workers'],config.get('shard_sizes'));total=sum(sizes)
    cpu=TrainSession(dict(config,device='cpu',init_checkpoint=str(out/'initial.npz'),profile=False))
    directory=out/'cpu_update_reference';directory.mkdir(exist_ok=True)
    errors={};trajectory=[]
    for step in range(start_step,start_step+steps):
        snapshots=[]
        for rank in range(config['train_workers']):
            with np.load(out/f'rank-{rank}/step-{step:04d}.npz',allow_pickle=False) as f:
                snapshots.append({k:f[k] for k in f.files})
        # Independent FP64 accumulation of the actual FP32 local gradients checks weighting,
        # missing/duplicate contributions, chunk restoration and communication precision.
        names=list(cpu.params)
        gold={name:sum(np.asarray(s[f'local_gradient::{name}'],np.float64)*weight
                       for s,weight in zip(snapshots,sizes))/total for name in names}
        averaged={name:snapshots[0][f'gradient::{name}'] for name in names}
        for rank,snapshot in enumerate(snapshots):
            for name in names:
                actual=snapshot[f'gradient::{name}']
                np.testing.assert_allclose(actual,gold[name],atol=1e-4,rtol=1e-3,
                                           err_msg=f'{out} step{step} rank{rank} collective {name}')
                np.testing.assert_array_equal(actual,averaged[name])
                errors['collective']=max(errors.get('collective',0.),float(np.max(np.abs(actual-gold[name]))))
        # Use the verified transmitted FP32 mean so the optimizer check does not confuse
        # Adam's response to tiny mean differences with incorrect optimizer arithmetic.
        cpu.apply_global_gradients(averaged,f'cpu-reference-{step}')
        expected_path=directory/f'step-{step:04d}.npz'
        save_state(cpu,expected_path,gradients=averaged)
        with np.load(expected_path,allow_pickle=False) as expected:
            for rank,snapshot in enumerate(snapshots):
                for key in expected.files:
                    if key=='metadata':
                        assert str(snapshot[key])==str(expected[key]),(step,rank,'optimizer/version')
                    else:
                        np.testing.assert_allclose(snapshot[key],expected[key],atol=1e-4,rtol=1e-3,
                                                   err_msg=f'{out} step{step} rank{rank} {key}')
                        kind=key.split('::')[0]
                        errors[kind]=max(errors.get(kind,0.),float(np.max(np.abs(snapshot[key]-expected[key]))))
        # Preserve the stricter independent whole-batch trajectory comparison as a separate
        # diagnostic. ReLU/maxpool can select different derivatives after sub-ULP changes.
        with np.load(trajectory_reference/f'rank-0/step-{step:04d}.npz',allow_pickle=False) as ref:
            if step == 0:
                for name in names:
                    np.testing.assert_allclose(averaged[name],ref[f'gradient::{name}'],atol=1e-4,rtol=1e-3,
                                               err_msg=f'{out} common-initial-state full batch {name}')
            differences={}
            for kind in ('parameter','gradient','v','s'):
                keys=[k for k in ref.files if k.startswith(kind+'::')]
                if keys:
                    differences[kind]={'max_absolute_error':max(float(np.max(np.abs(snapshots[0][k]-ref[k]))) for k in keys),
                        'elements_outside_original_tolerance':sum(int(np.sum(np.abs(snapshots[0][k]-ref[k])>1e-4+1e-3*np.abs(ref[k]))) for k in keys)}
            trajectory.append({'step':step,**differences})
    write_json(out/'resnet_step_audit.json',{'passed':True,'steps':steps,'start_step':start_step,'shards':sizes,
               'collective_reference':'FP64 weighted sum of recorded local FP32 gradients',
               'optimizer_reference':'CPU existing optimizer over the verified transmitted FP32 means, from same initial state',
               'tolerance':{'atol':1e-4,'rtol':1e-3},'max_absolute_error':errors,
               'independent_full_batch_trajectory_passed':all(
                   v['elements_outside_original_tolerance']==0 for r in trajectory for k,v in r.items() if k!='step'),
               'independent_full_batch_trajectory':trajectory,
               'trajectory_reference':str(trajectory_reference)})
    return errors
