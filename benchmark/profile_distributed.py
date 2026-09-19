"""Capture actual GPU kernels in PS/Ring worker processes, outside timings."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from benchmark.distributed_suite import task_config
from MyFlows.distributed.measurement import write_json


def run(command, log):
    with log.open('w',encoding='utf-8') as f:
        result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=480)
    if result.returncode:
        raise RuntimeError(f'profiler exit {result.returncode}: {log}')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='docs/experiments/semester_2026_fall/distributed_gpu/20260917')
    ap.add_argument('--tool',choices=('nsys','ncu'),default='nsys')
    ap.add_argument('--task',choices=('mnist_mlp','donkey_cnn','donkey_resnet18'),default='donkey_resnet18')
    args=ap.parse_args();root=Path(args.root).resolve()
    executable=(Path('C:/Program Files/NVIDIA Corporation/Nsight Systems 2026.1.3/target-windows-x64/nsys.exe') if args.tool=='nsys'
                else Path('C:/Program Files/NVIDIA Corporation/Nsight Compute 2026.2.1/target/windows-desktop-win7-x64/ncu.exe'))
    summaries=[]
    for mode in ('ps','ring'):
        original=root/'profiles'/f'{args.task}-{args.tool}-{mode}'
        folder=original
        attempt=1
        while folder.exists():
            attempt+=1
            folder=original.with_name(original.name+f'-attempt{attempt}')
        folder.mkdir(parents=True,exist_ok=False)
        config=task_config(root,args.task)
        config.update(mode=mode,transport='grpc_proto',train_workers=2,global_batch=8,epochs=1,steps=5,
                      evaluate=False,final_test=False,monitor=False,profile=False)
        write_json(folder/'config.json',config)
        workload=[sys.executable,'-X','utf8','-m','benchmark.distributed_experiment','--config',str(folder/'config.json'),'--out',str(folder/'workload')]
        prefix=folder/'capture'
        if args.tool=='nsys':
            command=[str(executable),'profile','--trace=cuda,nvtx,cublas','--sample=none','--cpuctxsw=none','--wait=all',f'--output={prefix}',*workload]
            report=prefix.with_suffix('.nsys-rep')
        else:
            kernel='regex:.*gemm.*' if args.task=='mnist_mlp' else 'im2col_forward'
            command=[str(executable),'--config-file','off','--target-processes','all','--set','full','--kernel-name',kernel,
                     '--launch-count','1','--export',str(prefix),*workload]
            report=prefix.with_suffix('.ncu-rep')
        write_json(folder/'command.json',command)
        print('PROFILE',args.tool,mode,flush=True)
        run(command,folder/'profiler.log')
        workload_result=json.loads((folder/'workload/results.json').read_text(encoding='utf-8'))
        if workload_result['status']!='passed' or workload_result['alive_pids'] or not workload_result['rank_state_consistent']:
            raise AssertionError('profiled training failed or ranks disagree')
        if not report.is_file() or report.stat().st_size==0:raise RuntimeError('missing profiler report')
        if args.tool=='nsys':
            run([str(executable),'stats','--report','cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_trace','--format','csv',
                 '--output',str(folder/'summary'),str(report)],folder/'stats.log')
            summary=(folder/'summary_cuda_gpu_kern_sum.csv').read_text(encoding='utf-8-sig')
            required=['gemm'] if args.task=='mnist_mlp' else ['im2col_forward','col2im_backward','maxpool2d_forward_native','maxpool2d_backward_native']
        else:
            run([str(executable),'--import',str(report),'--page','raw','--csv'],folder/'metrics.csv')
            summary=(folder/'metrics.csv').read_text(encoding='utf-8-sig')
            required=['gemm'] if args.task=='mnist_mlp' else ['im2col_forward']
        for name in required:
            if name not in summary:raise AssertionError('kernel absent: '+name)
        summaries.append({'task':args.task,'mode':mode,'tool':args.tool,'report':str(report.relative_to(root)),
                          'required_kernels':required,'passed':True})
        write_json(root/f'{args.tool}_checks.json',summaries)
        print('PROFILE PASS',mode,flush=True)


if __name__=='__main__':main()
