"""Generate Chinese course tables and scientific plots from retained raw runs."""

import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from benchmark.distributed_compare import MATRIX
from benchmark.distributed_experiment import read_jsonl
from benchmark.distributed_suite import resolve_run


LABELS={'single-1':'单进程', 'ps-json-2':'PS JSON 2', 'ps-json-4':'PS JSON 4',
        'ps-grpc-2':'PS gRPC 2','ps-grpc-4':'PS gRPC 4', 'ring-grpc-2':'Ring gRPC 2','ring-grpc-4':'Ring gRPC 4'}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def export_course_table(root, task, title, rows, kind):
    selected=[r for r in rows if ('ring' not in r['config_id'] if kind=='ps' else 'json' not in r['config_id'])]
    quality_label='验证准确率' if task=='mnist_mlp' else '验证 angle MAE'
    headers=['配置','epoch 总时间/s','计算图/s\n主机耗时','梯度同步/s\n含聚合与等待',
             '正文总量/MiB' if kind=='ps' else '加速比',quality_label]
    cells=[]
    for row in selected:
        quality=f"{row['val_accuracy']:.2%}" if task=='mnist_mlp' else f"{row['val_angle_mae']:.5f}"
        fifth=f"{(row['request_bytes']+row['response_bytes'])/2**20:,.2f}" if kind=='ps' else f"{row['speedup']:.3f}"
        cells.append([LABELS[row['config_id']],f"{row['epoch_s_median']:.3f}",f"{row['compute_wall_s']:.3f}",
                      f"{row['gradient_sync_s']:.3f}",fifth,quality])
    fig,ax=plt.subplots(figsize=(12,4.1));ax.axis('off')
    table=ax.table(cellText=cells,colLabels=headers,cellLoc='center',colLoc='center',
                   colWidths=[.22,.16,.15,.15,.16,.16],bbox=[0,.10,1,.78])
    table.auto_set_font_size(False);table.set_fontsize(11)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('white')
        if r==0:
            cell.set_facecolor('#4472c4');cell.get_text().set_color('white');cell.get_text().set_weight('bold')
        else:
            cell.set_facecolor('#e8edf7' if r%2 else '#d5def0')
    suffix='PS 协议对比' if kind=='ps' else 'PS / Ring 对比 · 全部 gRPC+protobuf'
    ax.set_title(title+'：'+suffix,loc='left',fontsize=16,pad=12,color='#1565a5')
    fig.text(.035,.048,'性能为 3 次完整 epoch 的中位数；质量来自完整收敛训练。阶段取各 rank 累计的最大值；所有 Worker 共享一张 GPU。',fontsize=8,color='#555555')
    model_note='现有 ResNet18：base_width=16，固定训练集校准的 BN 统计，120×160 RGB；道路指标为 angle 回归误差。' if task=='donkey_resnet18' else 'MNIST：784→64→10 MLP，global batch=128；GPU 为 RTX 4060 Laptop。'
    fig.text(.035,.013,model_note,fontsize=8,color='#555555')
    fig.tight_layout(rect=[.02,.06,.98,1]);fig.savefig(root/f'{task}-{kind}-table.png');plt.close(fig)


def resource_stats(directory):
    ranges=[(r['start_s'],r['end_s']) for r in read_jsonl(directory/'rank-0/epochs.jsonl')]
    file=directory/'resources.jsonl'
    rows=read_jsonl(file) if file.exists() else []
    samples=[r for r in rows if any(start<=r['monotonic_s']<=end for start,end in ranges)]
    result={'resource_samples':len(samples)}
    for key in ('gpu_utilization_pct','gpu_memory_mib','gpu_temperature_c','gpu_power_w','cpu_utilization_pct'):
        values=[r[key] for r in samples if key in r]
        result[key+'_mean']=float(np.mean(values)) if values else None
        result[key+'_max']=max(values) if values else None
    return result


def export_epoch_metrics(root, tasks):
    """Retain identifiers and measurement boundaries beside every epoch value."""
    rows=[]
    for task in tasks:
        for phase in ('convergence','performance/repeat-1','performance/repeat-2','performance/repeat-3'):
            baseline=load(resolve_run(root,f'{task}/{phase}/single-1')/'results.json')['epochs']
            for cid,mode,transport,n in MATRIX:
                directory=resolve_run(root,f'{task}/{phase}/{cid}')
                config=load(directory/'config.json'); result=load(directory/'results.json')
                data=load(directory/'data_manifest.json'); initial=load(directory/'rank-0/initial.json')
                rank_epochs=[read_jsonl(directory/f'rank-{rank}/epochs.jsonl') for rank in range(n)]
                steps=[r for rank in range(n) for r in read_jsonl(directory/f'rank-{rank}/steps.jsonl')]
                for ep in result['epochs']:
                    e=ep['epoch']; current=[group[e] for group in rank_epochs]
                    epoch_steps=[r for r in steps if r['epoch']==e]
                    speedup=baseline[e]['epoch_train_wall_s']/ep['epoch_train_wall_s']
                    checkpoint=directory/f'epoch-{e:03d}.npz'
                    row={'run_id':directory.relative_to(root).as_posix(),'task':task,'phase':phase,
                         'config_id':cid,'mode':mode,'transport':transport,'workers':n,
                         'seed':config['seed'],'global_batch':config['global_batch'],
                         'optimizer':initial['optimizer']['impl_path'],'learning_rate':config['learning_rate'],
                         'update_device':config['device'],'split_fingerprint':data['split_fingerprint'],
                         'initial_state_digest':initial['digest'],
                         'stage_aggregation':'maximum of per-rank epoch sums',
                         'checkpoint':checkpoint.relative_to(root).as_posix() if checkpoint.exists() else '',
                         **ep,'graph_compute_s':max(r['forward_gpu_s']+r['backward_gpu_s'] for r in current),
                         'S':speedup,'E':speedup/n,
                         'train_loss':sum(r['loss']*r['n_samples'] for r in epoch_steps)/sum(r['n_samples'] for r in epoch_steps),
                         'sync_update_s':max(r['sync_update_s'] for r in current) if all('sync_update_s' in r for r in current) else None,
                         'sync_update_measured':all('sync_update_s' in r for r in current),
                         'pure_network_transfer_s':None}
                    rows.append(row)
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with (root/'epoch_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='docs/experiments/semester_2026_fall/distributed_gpu/20260917')
    ap.add_argument('--tasks',nargs='+',choices=('mnist_mlp','donkey_cnn','donkey_resnet18'))
    args=ap.parse_args();root=Path(args.root).resolve()
    scope=load(root/'execution_scope.json') if (root/'execution_scope.json').exists() else {}
    tasks=args.tasks or [t for t in scope.get('tasks',['mnist_mlp']) if t not in ('synthetic','donkey_cnn')]
    export_epoch_metrics(root,tasks)
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'], 'axes.unicode_minus':False,
                         'figure.dpi':150, 'savefig.dpi':180, 'axes.spines.top':False,'axes.spines.right':False})
    all_rows=[]
    sections=[]
    for task,title in (('mnist_mlp','MNIST MLP'),('donkey_cnn','道路小型 CNN'),('donkey_resnet18','DonkeyCar ResNet18')):
        if task not in tasks:
            continue
        rows=[]
        baseline=None
        for cid,mode,transport,n in MATRIX:
            repeated=[]
            for repeat in (1,2,3):
                directory=resolve_run(root,f'{task}/performance/repeat-{repeat}/{cid}')
                result=load(directory/'results.json')
                assert result['status']=='passed'
                ep=result['epochs'][0]
                repeated.append((directory,ep))
            epochs=[e for _,e in repeated]
            time_med=float(np.median([e['epoch_train_wall_s'] for e in epochs]))
            if cid=='single-1':baseline=time_med
            quality_dir=resolve_run(root,f'{task}/convergence/{cid}')
            quality=load(quality_dir/'results.json')
            row={'task':task,'config_id':cid,'workers':n,'epoch_s_median':time_med,
                 'epoch_s_min':min(e['epoch_train_wall_s'] for e in epochs),
                 'epoch_s_max':max(e['epoch_train_wall_s'] for e in epochs),
                 'epoch_s_p95':float(np.percentile([e['epoch_train_wall_s'] for e in epochs],95)),
                 'speedup':baseline/time_med,'efficiency':baseline/time_med/n}
            for key in epochs[0]:
                if key in ('epoch','epoch_train_wall_s') or key.startswith('val_'):
                    continue
                values=[e.get(key) for e in epochs if e.get(key) is not None]
                row[key]=float(np.median(values)) if values else None
            row['sync_fraction']=row['gradient_sync_s']/time_med
            row['val_accuracy']=quality['final']['val'].get('accuracy')
            row['val_angle_mae']=quality['final']['val'].get('angle_mae')
            row['test_accuracy']=quality['final'].get('test',{}).get('accuracy')
            row['test_angle_mae']=quality['final'].get('test',{}).get('angle_mae')
            row['quality_run']=quality_dir.relative_to(root).as_posix()
            row['performance_runs']=';'.join(str(p.relative_to(root)) for p,_ in repeated)
            resources=[resource_stats(p) for p,_ in repeated]
            for key in resources[0]:
                vals=[r[key] for r in resources if r[key] is not None]
                if key.endswith('_max'):
                    row[key]=max(vals) if vals else None
                elif key=='resource_samples':
                    row[key]=sum(vals)
                else:
                    row[key]=float(np.mean(vals)) if vals else None
            rows.append(row);all_rows.append(row)
        export_course_table(root,task,title,rows,'ps')
        export_course_table(root,task,title,rows,'ring')
        quality_label='验证准确率' if task=='mnist_mlp' else '验证 angle MAE'
        text=[f'## {title}', '', '### PS 与 Ring：统一 gRPC/protobuf', '', '|配置|epoch 时间/s|计算/s|梯度同步/s|加速比|效率|'+quality_label+'|',
              '|---|---:|---:|---:|---:|---:|---:|']
        for r in [r for r in rows if 'json' not in r['config_id']]:
            value=f"{100*r['val_accuracy']:.2f}%" if task=='mnist_mlp' else f"{r['val_angle_mae']:.6f}"
            text.append(f"|{LABELS[r['config_id']]}|{r['epoch_s_median']:.3f}|{r['compute_wall_s']:.3f}|{r['gradient_sync_s']:.3f}|{r['speedup']:.3f}|{r['efficiency']:.3f}|{value}|")
        text += ['',f'课件表格图片：[PS 协议表]({task}-ps-table.png)、[统一 gRPC 的 PS/Ring 表]({task}-ring-table.png)。',
                 '计算列为前向加反向的主机耗时；CUDA Event 对应数据在阶段明细中另列。梯度同步包含聚合和等待。']
        text += ['', '### PS 传输协议对比（对应第一张课件表）', '',
                 '|配置|epoch 时间/s|计算/s|通信/s|消息正文总字节/epoch|'+quality_label+'|',
                 '|---|---:|---:|---:|---:|---:|']
        for cid in ('single-1','ps-json-2','ps-grpc-2','ps-json-4','ps-grpc-4'):
            r=next(r for r in rows if r['config_id']==cid)
            value=f"{r['val_accuracy']:.2%}" if task=='mnist_mlp' else f"{r['val_angle_mae']:.6f}"
            text.append(f"|{LABELS[cid]}|{r['epoch_s_median']:.3f}|{r['compute_wall_s']:.3f}|{r['gradient_sync_s']:.3f}|{r['request_bytes']+r['response_bytes']:.0f}|{value}|")
        text += ['', '消息正文为请求与响应的有向发送总量，每条消息仅计一次；不包括 TCP/IP、HTTP/2 头。正文包括梯度及控制消息。', '', '### 质量与阶段明细', '']
        text += ['', '|配置|梯度 D2H/s|均值送入设备/s|GPU 更新/s|摘要及更新确认/s|反向后至放行/s|',
                 '|---|---:|---:|---:|---:|---:|']
        for r in rows:
            text.append(f"|{LABELS[r['config_id']]}|{r['gradient_d2h_s']:.3f}|{r['gradient_to_device_wall_s']:.3f}|{r['optimizer_gpu_s']:.3f}|{r['update_confirm_s']:.3f}|{r['sync_update_s']:.3f}|")
        text += ['', '反向后至放行为单独主机时间戳测量，包含指标处理、梯度拷贝/同步、更新及确认。均值送入设备一列在单进程中为 GPU 内部复制，分布式中含 H2D。摘要时间包含在更新确认列中。', '']
        if task=='mnist_mlp':
            text += ['', '|配置|验证准确率|官方测试准确率|与单进程验证差/百分点|', '|---|---:|---:|---:|']
            for r in rows:
                text.append(f"|{LABELS[r['config_id']]}|{r['val_accuracy']:.2%}|{r['test_accuracy']:.2%}|{100*(r['val_accuracy']-rows[0]['val_accuracy']):+.3f}|")
        else:
            text += ['', '|配置|验证 angle MAE|测试 angle MAE|测试 angle RMSE|测试 throttle MAE|', '|---|---:|---:|---:|---:|']
            for r in rows:
                quality=load(resolve_run(root,f"{task}/convergence/{r['config_id']}")/'results.json')['final']
                test=quality['test']
                text.append(f"|{LABELS[r['config_id']]}|{r['val_angle_mae']:.6f}|{test['angle_mae']:.6f}|{test['angle_rmse']:.6f}|{test['throttle_mae']:.6f}|")
            text += ['', '道路数据是 angle/throttle 连续值回归，课件的“准确率”列用有明确意义的 MAE/RMSE 替代。throttle 标签恒为 0.5，不代表具备变速决策能力。']
        text += ['', '|配置|forward GPU/s|backward GPU/s|update GPU/s|数据准备/s|samples/s|单步中位/ms|单步P95/ms|',
                 '|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            text.append(f"|{LABELS[r['config_id']]}|{r['forward_gpu_s']:.3f}|{r['backward_gpu_s']:.3f}|{r['optimizer_gpu_s']:.3f}|{r['data_s']:.3f}|{r['samples_per_s']:.0f}|{1000*r['rank_step_median_s']:.3f}|{1000*r['rank_step_p95_s']:.3f}|")
        text += ['', '|配置|GPU利用率均值/%|采样显存峰值/MiB|温度峰值/℃|功耗峰值/W|CPU利用率均值/%|有效采样点|',
                 '|---|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            values=[r[key] for key in ('gpu_utilization_pct_mean','gpu_memory_mib_max','gpu_temperature_c_max','gpu_power_w_max','cpu_utilization_pct_mean')]
            text.append('|'+LABELS[r['config_id']]+'|'+'|'.join(f'{v:.1f}' if v is not None else '未采到' for v in values)+f"|{r['resource_samples']}|")
        text += ['', '资源均值为三次 run 各自有效样本均值的平均；峰值为三次 run 中的最大采样值。短 epoch 采样点少，不能据此精确估计持续利用率。', '',
                 '|配置|请求正文 MiB/epoch|响应正文 MiB/epoch|训练梯度上行 MiB/epoch|同步占 epoch 比例|',
                 '|---|---:|---:|---:|---:|']
        for r in rows:
            text.append(f"|{LABELS[r['config_id']]}|{r['request_bytes']/2**20:.2f}|{r['response_bytes']/2**20:.2f}|{r['outgoing_gradient_bytes']/2**20:.2f}|{r['sync_fraction']:.1%}|")
        text += ['', '正文为各 Worker 的实际请求/响应累计值再取三次中位数，包含控制消息；梯度上行为 PS Worker→Server 或 Ring 各有向链路发送的 FP32 数组字节，PS 返回梯度不计入该列。']
        fig,(ax,zoom)=plt.subplots(1,2,figsize=(11,4.5),gridspec_kw={'width_ratios':[1.25,1]})
        ax.plot([1,2,4],[1,2,4],'--',color='#999999',label='理想线性 S=N（共享单卡参考）')
        for prefix,label,color in [('ps-grpc','PS gRPC/protobuf','#2563eb'),('ring-grpc','Ring gRPC/protobuf','#059669')]:
            seq=[rows[0]]+[next(r for r in rows if r['config_id']==f'{prefix}-{n}') for n in (2,4)]
            ys=[r['speedup'] for r in seq]
            low=[y-baseline/r['epoch_s_max'] for y,r in zip(ys,seq)]
            high=[baseline/r['epoch_s_min']-y for y,r in zip(ys,seq)]
            ax.errorbar([1,2,4],ys,yerr=[low,high],marker='o',capsize=4,label=label,color=color)
            zoom.errorbar([2,4],ys[1:],yerr=[low[1:],high[1:]],marker='o',capsize=4,label=label,color=color)
        ax.set(xlabel='训练 Worker 数',ylabel='加速比（单进程时间 / 当前时间）',title=title+'：相同全局 batch，共享一张 GPU',xticks=[1,2,4],ylim=(0,4.3))
        zoom.set(xlabel='训练 Worker 数',ylabel='加速比',title='N=2/4 实测范围放大',xticks=[2,4],ylim=(0,None))
        ax.grid(alpha=.2);zoom.grid(alpha=.2);ax.legend(fontsize=8);zoom.legend(fontsize=8)
        fig.tight_layout();fig.savefig(root/f'{task}-speedup.png');plt.close(fig)
        text+=['',f'![{title} 加速比]({task}-speedup.png)','',
               '误差线为 3 次独立进程测量的最小/最大耗时换算范围；N=1 为共同单进程基线。']
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for cid,_,_,_ in MATRIX:
            result=load(resolve_run(root,f'{task}/convergence/{cid}')/'results.json')
            ep=result['epochs']
            axes[0].plot([e['epoch']+1 for e in ep],[e['val_loss'] for e in ep],label=LABELS[cid],linewidth=1.1)
            metric='val_accuracy' if task=='mnist_mlp' else 'val_angle_mae'
            axes[1].plot([e['epoch']+1 for e in ep],[e[metric] for e in ep],label=LABELS[cid],linewidth=1.1)
        axes[0].set(xlabel='epoch',ylabel='验证 loss',title=title+' 收敛')
        axes[1].set(xlabel='epoch',ylabel=quality_label,title='相同数据、初值和训练预算')
        for ax in axes:
            ax.grid(alpha=.2);ax.set_xticks(range(1,len(ep)+1))
        axes[1].legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(root/f'{task}-convergence.png');plt.close(fig)
        text+=['',f'![{title} 收敛]({task}-convergence.png)','']
        fastest=min(rows[1:],key=lambda r:r['epoch_s_median'])
        text += [f"本组分布式配置中耗时最短的是 {LABELS[fastest['config_id']]}，加速比为 {fastest['speedup']:.3f}。",
                 f"单进程完整 epoch 中位数为 {baseline:.3f} s；分布式同步包含主机梯度规约、编解码、回环 RPC 及等待，",
                 '且所有 Worker 争用同一张 GPU。Worker 数增加不增加 GPU 算力；消息轮次、Python 处理及更新确认会产生额外开销。', '']
        text += ['### 同 Worker 数的瓶颈对照','']
        for n in (2,4):
            ps=next(r for r in rows if r['config_id']==f'ps-grpc-{n}')
            ring=next(r for r in rows if r['config_id']==f'ring-grpc-{n}')
            js=next(r for r in rows if r['config_id']==f'ps-json-{n}')
            pb=ps['request_bytes']+ps['response_bytes'];jb=js['request_bytes']+js['response_bytes']
            text.append(f"- {n} Worker：PS gRPC / Ring gRPC 的 epoch 耗时比为 {ps['epoch_s_median']/ring['epoch_s_median']:.2f}；"
                        f"各自梯度同步占 epoch 的 {ps['sync_fraction']:.1%} / {ring['sync_fraction']:.1%}，"
                        f"更新确认累计为 {ps['update_confirm_s']:.3f} / {ring['update_confirm_s']:.3f} s。"
                        f"PS JSON / PS gRPC 的耗时比为 {js['epoch_s_median']/ps['epoch_s_median']:.2f}，消息正文总量比为 {jb/pb:.2f}。")
        text += ['', '以上比例是当前实现的实测结果。PS 的通用消息转换、轮询和服务器处理，以及 Ring 的分块交换轮次均包含在同步耗时中；没有独立消融数据来把各项差距全部归因于某一种机制。','']
        sections.extend(text)
    with (root/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    numeric=load(root/'numeric_checks.json')
    faults=load(root/'fault_checks.json')
    quality=load(root/'quality_checks.json')
    quality=[q for q in quality if q['task'] in tasks and Path(q['run'])==resolve_run(root,f"{q['task']}/convergence/{q['config']}")]
    states=load(root/'state_checks.json')
    volumes=load(root/'volume_checks.json')
    flows=load(root/'ring_chunk_flow.json')
    intro=['# MNIST 与 ResNet18 道路任务：PS / Ring 实测对比','',
           '根据用户恢复任务后的要求，正式比较覆盖 MNIST MLP 与现有 ResNet18 的 DonkeyCar 道路回归。PS 协议表包含 Socket/JSON 与 gRPC/protobuf；PS/Ring 架构表及加速比图统一使用 gRPC/protobuf。','',
           '本报告使用当前自研框架，在单机回环网络中运行独立 Worker，所有训练 Worker 共享一张 RTX 4060 Laptop GPU。',
           'PS 使用独立 CPU 聚合进程，Worker 在 GPU 更新参数；Ring 使用真实相邻 gRPC 传递，完成 ScatterReduce 与 AllGather。',
           'MNIST 使用 GPU 上的 CuPy 数组与矩阵乘法；ResNet18 的 Conv/Pool 使用现有原生 cuda_native_cublas。两任务沿用项目现有 Adam 更新规则。','',
           '**验收边界：ResNet 的独立整批训练轨迹与分片轨迹未通过原 Spec 的逐步梯度等价门槛。本文保留该失败，并分别报告已验证的梯度归约、CPU/GPU 更新一致性和完整训练质量，不将本报告称为全部 Spec 条款无条件通过。**','',
           '## 课件要求与证据','',
           '|来源|要求|保留证据|','|---|---|---|',
           '|PS 课件第 2、4 页|MNIST/MLP、道路 CNN；单机、2/4 Worker；Socket/JSON 与 gRPC|两任务各 7 个完整训练配置，*/convergence/*|',
           '|PS 课件第 43 页|心跳、掉线超时处理|fault_checks.json；PS 独立 watchdog，超时中止同步训练|',
           '|Ring 课件第 4、39 页|epoch 总耗时、计算、梯度同步、加速比、准确率|comparison.csv、每 run 的 epochs.csv、加速比与收敛图|',
           '|Ring 课件第 4、41 页|完整 Ring、计算正确性、固定格式归档|numeric_checks.json、逐 rank/step 的参数/梯度/Adam 状态 NPZ、messages.jsonl|',
           '|综合项目第 10 页|GPU/CPU 资源、forward/backward/update、加载、吞吐量|resources.jsonl、rank-*/steps.jsonl、epochs.csv|','',
           f"数值与更新检查保留 {len(numeric)} 组配置，覆盖 synthetic、MNIST、小型道路 CNN 和 ResNet18，MBGD/Adam、N=1/2/4 和非均匀分片。ResNet18 使用下文单列的归约/更新参考，不混称为独立整批轨迹全部等价。",
           f"故障/重复消息场景通过 {sum(r['passed'] for r in faults)}/{len(faults)}；训练质量门槛通过 {sum(r['quality_passed'] for r in quality)}/{len(quality)}。",'',
           f"带非空 Adam 状态的恢复对照通过 {len(states)}/{len(states)}；N=1/2/4/5 的分块流向与非整除补零检查通过 {len(flows)}/{len(flows)}；实际梯度字节量通过 {len(volumes)}/{len(volumes)}。",
           '数值/恢复矩阵中保留此前小型道路 CNN 的检查；它与本轮 ResNet18 正式结果单独存放。所有记录分别见 [数值](numeric_checks.json)、[恢复](state_checks.json)、[故障](fault_checks.json)、[分块](ring_chunk_flow.json)、[通信量](volume_checks.json)。','',
           '## 固定配置与测量口径','',
           '- MNIST：官方 60,000 训练部分划分为 50,000/10,000；官方 10,000 测试集仅在最终评估使用。784→64→10，global batch=128，Adam lr=0.001，10 epochs。',
           '- 每个 epoch 先排列全部 50,000 个训练索引，再统一丢弃尾部 80 个，共 390 steps、49,920 samples；不同 Worker 数拆分同一个 global batch。',
           '- 质量目标为验证 accuracy ≥95%，与单进程差 ≤1 个百分点。这是工作区 Spec 的项目门槛，不是课件给定的精确数值。',
           '- 各配置使用相同 seed=0、FP32、数据顺序和初始参数。性能数据为从同一初始 checkpoint 出发的完整第 1 个 epoch，独立进程重复 3 次，轮换配置顺序；所有 GPU 实验串行。',
           '- epoch 总时间排除进程启动、数据准备、首次编译预热、评估与保存；逐 Worker 的计算/同步累计后取最大值，不能把各列相加当作端到端耗时。',
           '- CUDA Event 测量 forward/backward/update；另保留主机 wall time。梯度 D2H、H2D、摘要及更新确认分列。GPU/CPU 以约 1 秒周期采样，显存峰值为采样峰值，并非瞬时精确峰值。',
           '- 利用率/显存/温度/功耗为整卡 GPU 观测，CPU 利用率为整机观测；没有扣除桌面与其他后台活动。进程树 RSS 另保留在 resources.jsonl。',
           '- 所有主表 run 启用相同 CUDA Event 同步、逐步记录和资源监控；表中是带测量开销的时间。监控开关对照只衡量资源采样，不能代表全部仪表开销。',
           '- Socket 的解码未单独计时，包含在 RPC 观测中；日志 decode_s=0 是占位值。编码、RPC、规约与等待均包含在梯度同步总时间，不称为纯网络传输时间。',
           '- 消息量为实际 JSON/protobuf 正文长度；Socket 长度前缀单列。HTTP/2、TCP/IP 头、重传未抓包测量，因此不称为实际网卡总流量。',
           '- 性能 run 与收敛 run 分别保存；表中的准确率/MAE来自完整收敛 run，耗时来自 3 次性能 run，具体 run_id 在 comparison.csv。','',
           '完整逐 epoch 数据见 [epoch_metrics.csv](epoch_metrics.csv)，包含 run/config、数据指纹、初值摘要、优化器、样本数、阶段时间、S/E 与质量。旧 MNIST 收敛 run 没有反向后至放行的独立时间戳，该字段留空并标记未测；三次正式性能重复均记录此字段。纯线路传输时间也留空，不用 0 冒充实测值。','',
           '## 通信量口径','',
           '令 N 为 Worker 数，M 为一个完整 FP32 梯度的字节数（PS 不计入 N）。PS 中心上行接收 NM、下行发送 NM；集群有向发送总量 2NM。',
           'Ring 每个 rank 单向发送/接收均为 2(N−1)M/N，集群有向发送总量为 2(N−1)M。单 rank 的量趋近 2M，集群总量仍随 N 增长。',
           '实际 Ring 尾部补零，用 M′=4N ceil(D/N) 代入。以上只计训练梯度；初始化、确认、心跳等控制消息由正文计数另行记录。','']
    frozen=load(root/'frozen_task_configs.json')['donkey_resnet18']
    resnet_model=load(resolve_run(root,'donkey_resnet18/convergence/single-1')/'rank-0/initial.json')['model']
    resnet_details=['## ResNet18 配置与 BN 控制','',
        f"复用 `{resnet_model['architecture']}`，8 个残差块，base_width={resnet_model['base_width']}，各 stage 为 {resnet_model['stage_widths']}，参数量 {resnet_model['parameters']:,}。这是现有模型的宽度配置，不是默认宽度 64 的 11,182,338 参数版本。",
        f"图像 120×160 RGB；8,000/1,000/1,000 按来源编号分组划分；global batch={frozen['global_batch']}，Adam lr={frozen['learning_rate']}，{frozen['epochs']} epochs。每 epoch 先全量排列再丢弃尾部 {8000%frozen['global_batch']} 张，使用 {8000//frozen['global_batch']*frozen['global_batch']} 张。",
        '仅用 512 张训练图校准 BN running mean/variance，随后固定；gamma/beta 及全部卷积/FC 权重保持可训练。所有配置复用同一 BN 文件，checkpoint 与参数一致性摘要包含 BN buffers。没有使用本地 shard 的训练态 BN，也未声称实现 SyncBatchNorm。',
        '预检仅访问训练/验证集，测试集在正式训练结束时评估。门槛：训练 loss 相对初始化下降至少 30%；验证 angle MAE 优于常数预测至少 10%；与单进程差 ≤max(0.001, 单进程 MAE 的 5%)。',
        '配置与预检证据：[冻结配置](frozen_task_configs.json)、[选择记录](resnet_preflight/freeze_decision.json)。','']
    resnet_checks=[r for r in numeric if 'donkey_resnet18' in r['run']]
    worst_collective=max(r['max_absolute_error'].get('collective',0.) for r in resnet_checks)
    worst_update=max(r['max_absolute_error'].get('parameter',0.) for r in resnet_checks)
    resnet_details += ['### ResNet 数值判定范围', '',
        '严格的“独立整批单进程轨迹 vs 分片轨迹，每一步梯度逐元素等价”在 ResNet 上出现超差，未将其标为通过。首个双 Worker MBGD 案例在第 4 步参数最大差约 4.3e-7，但梯度差约 1.4e-3；重算记录了 ReLU 掩码和 MaxPool argmax 的变化。某些相同权重下的不同 batch 形状也出现少量分支变化。详见 [保留诊断](resnet_gradient_diagnosis.json)。',
        '原 synthetic/MNIST/小型 CNN 保持原连续 20 步独立轨迹对照。新增 ResNet 检查分为：共同初始点的整批梯度对照；每一步实际局部 FP32 梯度的独立 FP64 加权求和；使用已经核验的传输均值，从同一初始状态执行现有 CPU 优化器，比较全部 rank 的 GPU 参数、Adam v/s、版本和 BN buffers。各项仍使用 atol=1e-4、rtol=1e-3；同时保留独立整批轨迹的逐步误差，未放宽容差后把原失败改写为成功。',
        f"新增 ResNet {len(resnet_checks)} 组逐步检查的归约最大绝对误差为 {worst_collective:.3g}，CPU/GPU 参数更新最大绝对误差为 {worst_update:.3g}。每组原始局部/全局梯度、CPU 参考 checkpoint 及 resnet_step_audit.json 均留档。完整训练的质量差异另按固定验证门槛判断。", '']
    extras=['## Profiler、监控开销和归档核验','']
    for tool in ('nsys','ncu'):
        for check in load(root/f'{tool}_checks.json'):
            report_path=check['report'].replace('\\','/')
            extras.append(f"- {tool} / {check['mode']}：已捕获 {', '.join(check['required_kernels'])}；[原始报告]({report_path})。")
    extras += ['', 'Profiler 使用独立的短训练进程，包含准备/预热，不计入性能表；Nsight Compute 抽取首个匹配 kernel 作代表性诊断。', '',
               '|Nsight Systems / 模式|原生 kernel|调用次数|累计 GPU 时间/ms|kernel 时间占比|',
               '|---|---|---:|---:|---:|']
    for check in load(root/'nsys_checks.json'):
        folder=(root/check['report']).parent
        with (folder/'summary_cuda_gpu_kern_sum.csv').open(encoding='utf-8-sig',newline='') as stream:
            kernels=list(csv.DictReader(stream))
        for kernel in kernels:
            if kernel['Name'] in check['required_kernels']:
                extras.append(f"|{check['mode']}|{kernel['Name']}|{kernel['Instances']}|{float(kernel['Total Time (ns)'])/1e6:.3f}|{kernel['Time (%)']}%|")
    extras += ['', '|Nsight Compute / 模式|采样 kernel|耗时/μs|SM 吞吐率/峰值|实际占用率|寄存器/线程|',
               '|---|---|---:|---:|---:|---:|']
    for check in load(root/'ncu_checks.json'):
        with ((root/check['report']).parent/'metrics.csv').open(encoding='utf-8-sig',newline='') as stream:
            kernels=list(csv.DictReader(stream))
        units=kernels[0]
        assert units['gpu__time_duration.sum']=='us'
        for kernel in kernels[1:]:
            extras.append(f"|{check['mode']}|{kernel['Kernel Name']}|{float(kernel['gpu__time_duration.sum']):.3f}|"
                          f"{float(kernel['sm__throughput.avg.pct_of_peak_sustained_elapsed']):.2f}%|"
                          f"{float(kernel['sm__warps_active.avg.pct_of_peak_sustained_active']):.2f}%|{kernel['launch__registers_per_thread']}|")
    extras += ['', '两种模式均记录到完整的原生 Conv/Pool 前后向路径。短追踪的 kernel 累计时间及单个 im2col 的占用率仅用于确认执行路径和说明该次采样；不能用这些预热/小 batch 数据代替上文完整 epoch 的瓶颈占比。', '',
               '|任务 / 配置|资源监控开/s|资源监控关/s|中位数差异|', '|---|---:|---:|---:|']
    for row in load(root/'monitor_overhead.json'):
        extras.append(f"|{row['task']} / {LABELS[row['config']]}|{row['on_s']:.3f}|{row['off_s']:.3f}|{row['relative_overhead']:+.1%}|")
    extras += ['', '开/关各 3 次独立进程，交替顺序，均测完整 epoch；原始值见 [monitor_overhead.json](monitor_overhead.json)。短时间差异包含调度和设备状态波动，负差值不能解释成监控具有加速作用。', '']
    for task in tasks:
        audit=load(root/f'{task}_artifact_audit.json')
        extras.append(f"- {task}：{audit['run_count']} 个正式 run 通过数据 SHA256、无交叉划分、相同初值、步数/样本数、GPU 参数/Adam 状态、rank 最终摘要和性能源码一致性核验；[记录]({task}_artifact_audit.json)。")
    for log in sorted((root/'suite_logs').glob('run_tests-*.log')):
        content=log.read_text(encoding='utf-8',errors='replace')
        counts=re.findall(r'Ran (\d+) tests in',content)
        if counts and content.rstrip().endswith('OK'):
            extras.append(f"- 全量回归 {counts[-1]} 项通过；[日志]({log.relative_to(root).as_posix()})。")
    extras += ['', '早期数值/恢复记录包含 Ring 控制消息仍用 Socket 的版本；新增 ResNet 复核和本轮正式 PS/Ring 矩阵均使用 gRPC/protobuf 控制消息，归档审计另行验证没有 Socket 帧。历史与新增结果保留各自源码指纹。','']
    ending=['## 可复现性与限制','',
            '每个 run 保留 config.json、manifest.json（两仓库 Git 状态、源码哈希、依赖）、data_manifest.json、results.json、epochs.csv、events.jsonl、resources.jsonl、逐 rank 的 step/epoch 日志及完整 checkpoint。',
            'MNIST 数组位于工作区 .codex/distributed-data-v1/mnist_mlp；数据 manifest 保存下载文件、数组和划分的 SHA256。',
            '共享单 GPU 的结果不能用来推断多 GPU/多机器网络扩展性。道路数据为 generated-road，编号分组并非经过验证的真实采集 session，离线回归指标不代表真实车辆闭环表现。','',
            '实现层限制：PS 的 gRPC 适配器目前在通用消息与 protobuf 之间经过 Python list，Ring 梯度块直接使用 bytes。统一通信协议后仍存在这类实现开销差异，因此比较结论针对本项目当前实现，不能把全部时间差归因于网络拓扑的理论优势。','',
            '完整复现：项目 .venv 下先运行 `python -m benchmark.prepare_distributed_data`，再运行 `python -m benchmark.complete_course_experiments --root <新目录>`。该入口依次执行 ResNet 预检、冻结配置、数值/恢复/故障检查、完整训练、三次性能重复、Profiler、监控对照、归档核验及报告生成。相同 root 再次执行会保留历史尝试并复用配置匹配的成功实验；不要同时启动两组 GPU 实验。']
    (root/'README.md').write_text('\n'.join(intro+resnet_details+sections+extras+ending)+'\n',encoding='utf-8')
    print(root/'README.md')


if __name__=='__main__':
    main()
