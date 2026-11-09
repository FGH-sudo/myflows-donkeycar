"""Recheck candidates for borderline or slow GEMMs through the production launcher against cuBLAS."""

import argparse
import json
import math
import os
from pathlib import Path
import random
import statistics

import numpy as np

from apps.autopilot.files import write_json
from apps.autopilot.provenance import code_state
from apps.common.gpu_lock import GpuLock
from benchmark.gemm_engines import _event_ms
from benchmark.resnet_before_after import ROOT
from MyFlows.core.device import set_device, xp
from MyFlows.ops.cuda_native import autotune, native


def qualify(row, repeats, out):
    layout = row["layout"]
    m, n, k = (row[key] for key in ("m", "n", "k"))
    autotune.selected_plan(layout, m, n, k)
    saved = json.loads(autotune.cache_file(layout,m,n,k).read_text(encoding="utf-8"))
    feasible = sorted((c for c in saved["candidates"] if not c.get("skip")),key=lambda c:c["median_ms"])
    choices = [(c["tile"],c["requested_slices"]) for c in feasible[:12]]
    for tile in native.tile_catalog():
        group = [c for c in feasible if c["tile"]==tile]
        for group2 in (group,[c for c in group if c["slices"]==1]):
            if group2:
                choice=(tile,group2[0]["requested_slices"])
                if choice not in choices:
                    choices.append(choice)
    rng=xp.random.RandomState(32)
    ta,tb=layout=="TN",layout=="NT"
    a=rng.standard_normal((k,m) if ta else (m,k),dtype=xp.float32)
    b=rng.standard_normal((n,k) if tb else (k,n),dtype=xp.float32)
    dest=xp.empty((m,n),dtype=xp.float32)
    call_reg=lambda: native.gemm(a,b,out=dest,trans_a=ta,trans_b=tb)
    call_blas=lambda: native.gemm(a,b,out=dest,trans_a=ta,trans_b=tb,engine="cublas")
    profiles=native._device_profile(xp.cuda.runtime.getDevice())
    key=(layout,m,n,k)
    original=profiles.get(key)
    ia=xp.asarray(np.linspace(0,m-1,min(8,m),dtype=int))
    ib=xp.asarray(np.linspace(0,n-1,min(8,n),dtype=int))
    exact=xp.asnumpy((a.T if ta else a)[ia]).astype(np.float64)@xp.asnumpy((b.T if tb else b)[:,ib]).astype(np.float64)
    rng_order=random.Random(67)
    records={p:{"tile":p[0],"requested_slices":p[1],"register_ms":[],"cublas_ms":[]} for p in choices}
    try:
        _event_ms(xp,call_blas,24)
        count=max(4,min(256,math.ceil(8./max(.001,_event_ms(xp,call_blas,8)))))
        for repeat in range(repeats):
            rng_order.shuffle(choices)
            for i,p in enumerate(choices):
                profiles[key]=p
                entry=records[p]
                order=(("register_ms",call_reg),("cublas_ms",call_blas))
                for label,fn in (order if (i+repeat)%2==0 else reversed(order)):
                    fn()
                    entry[label].append(float(_event_ms(xp,fn,count)))
                if repeat==0:
                    call_reg()
                    error=float(np.max(np.abs(xp.asnumpy(dest[ia[:,None],ib[None,:]])-exact))/max(1e-30,np.max(np.abs(exact))))
                    entry["sample_relative_vs_fp64"]=error
        for r in records.values():
            r["ratio"]=statistics.median(r["register_ms"])/statistics.median(r["cublas_ms"])
            r["paired_ratios"]=[x/y for x,y in zip(r["register_ms"],r["cublas_ms"])]
        ordered=sorted(records.values(),key=lambda r:r["ratio"])
        result={"layout":layout,"shape":[m,n,k],"batch_reps":count,"rounds":repeats,"selected":ordered[0],"finalists":ordered}
        write_json(out/f"{layout}-{m}x{n}x{k}.json",result)
        return result
    finally:
        if original is None:
            profiles.pop(key,None)
        else:
            profiles[key]=original


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--above",type=float,default=1.12)
    ap.add_argument("--rounds",type=int,default=7)
    args=ap.parse_args(argv)
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    rows=json.loads((args.gate/"gemms.json").read_text(encoding="utf-8"))
    set_device("cuda")
    old=os.environ.pop("MYFLOWS_GEMM_AUTOTUNE",None)
    try:
        with GpuLock(label="qualify measured register plans",run_dir=out):
            write_json(out/"manifest.json",{"source":code_state(ROOT),"above":args.above,"rounds":args.rounds})
            results=[]
            for row in sorted(rows,key=lambda r:-r["register_ms"]/r["cublas_ms"]):
                if row["register_ms"]/row["cublas_ms"] <= args.above:
                    continue
                result=qualify(row,args.rounds,out)
                results.append(result)
                win=result["selected"]
                print(result["layout"],result["shape"],win["tile"],win["requested_slices"],win["ratio"],flush=True)
                write_json(out/"summary.json",results)
                xp.get_default_memory_pool().free_all_blocks()
    finally:
        if old is not None:
            os.environ["MYFLOWS_GEMM_AUTOTUNE"]=old


if __name__=="__main__":
    main()
