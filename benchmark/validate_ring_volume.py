"""Audit measured gradient bytes and archive N=1/2/4/5 chunk flows."""

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.distributed_experiment import read_jsonl
from MyFlows.distributed.gradient_layout import padded_chunks, scatter_reduce_apply, allgather_apply
from MyFlows.distributed.measurement import write_json


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='docs/experiments/semester_2026_fall/distributed_gpu/20260917')
    root=Path(ap.parse_args().root).resolve()
    flow=[]
    for n in (1,2,4,5):
        for d in (1,7,12):
            inputs=[(np.arange(d,dtype=np.float32)+rank*10)*(rank+1) for rank in range(n)]
            states=[padded_chunks(v,n)[0].copy() for v in inputs]
            contributors=[[{rank} for _ in range(n)] for rank in range(n)]
            trace=[]
            for stage in ('SCATTER','GATHER'):
                for s in range(n-1):
                    messages=[]
                    for rank in range(n):
                        cid=(rank-s)%n if stage=='SCATTER' else (rank+1-s)%n
                        messages.append({'sender':rank,'receiver':(rank+1)%n,'chunk':cid,'values':states[rank][cid].tolist()})
                    new=[]
                    next_contributors=[[set(c) for c in rank] for rank in contributors]
                    for rank in range(n):
                        incoming=np.asarray(messages[(rank-1)%n]['values'],np.float32)
                        fn=scatter_reduce_apply if stage=='SCATTER' else allgather_apply
                        new.append(fn(states[rank],incoming,rank,s,n))
                        left=(rank-1)%n
                        cid=messages[left]['chunk']
                        if stage=='SCATTER':
                            assert not (contributors[rank][cid] & contributors[left][cid])
                            next_contributors[rank][cid] |= contributors[left][cid]
                        else:
                            next_contributors[rank][cid] = set(contributors[left][cid])
                    states=new
                    contributors=next_contributors
                    trace.append({'stage':stage,'round':s,'messages':messages})
            expected=np.sum(np.asarray(inputs,dtype=np.float64),axis=0)
            for state in states:
                np.testing.assert_allclose(state.reshape(-1)[:d],expected,atol=1e-8,rtol=1e-6)
            assert all(c==set(range(n)) for rank in contributors for c in rank)
            flow.append({'N':n,'D':d,'padding':n*((d+n-1)//n)-d,'passed':True,
                         'each_chunk_contains_each_rank_once':True,'flow':trace})
    write_json(root/'ring_chunk_flow.json',flow)
    checks=[]
    for path in sorted((root/'numeric').rglob('config.json')):
        cfg=json.loads(path.read_text(encoding='utf-8'))
        n=cfg['train_workers'];mode=cfg['mode']
        if mode=='single':continue
        out=path.parent
        result_file=out/'results.json'
        if not result_file.exists() or json.loads(result_file.read_text(encoding='utf-8')).get('status')!='passed':
            continue
        init=json.loads((out/'rank-0/initial.json').read_text(encoding='utf-8'))
        d=sum(int(np.prod(item['shape'])) for item in init['schema']);m=4*d
        expected=m if mode=='ps' else (2*(n-1)*((d+n-1)//n)*4 if n>1 else 0)
        observed=0
        for rank in range(n):
            for row in read_jsonl(out/f'rank-{rank}/steps.jsonl'):
                actual=int(row.get('outgoing_gradient_bytes',0))
                if actual!=expected:
                    raise AssertionError(f'{out} rank {rank} step {row["step"]}: {actual} != {expected}')
                observed+=actual
        checks.append({'run':str(out.relative_to(root)),'N':n,'D':d,'M':m,'rank_outgoing_per_step':expected,
                       'total_observed_outgoing':observed,'passed':True})
    write_json(root/'volume_checks.json',checks)
    print(f'algorithm cases={len(flow)}; measured byte cases={len(checks)}; all passed')


if __name__=='__main__':main()
