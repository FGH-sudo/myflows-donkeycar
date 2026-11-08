import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Alert, Select, Spin, Table } from 'antd'
import { api, type FrameworkRun } from '../api'
import ChartPanel from '../components/ChartPanel'
import EmptyState from '../components/EmptyState'
import { Panel, SectionHeader } from '../components/Panel'
import { baseGrid, baseLegend, modernTooltip } from './run/chartUtils'

const MODE = { fixed: '固定设备张量', e2e: '含数据与 H2D', quality: '训练质量' }
const NAME: Record<string, string> = { myflows: 'MyFlows', torch: 'PyTorch 确定性', 'torch-autotuned': 'PyTorch 自动调优' }

export default function FrameworkComparison() {
  const { data, isLoading, error } = useQuery({ queryKey: ['framework-comparisons'], queryFn: api.frameworks })
  const [selected, setSelected] = useState('resnet18-001')
  const [mode, setMode] = useState<FrameworkRun['mode']>('e2e')
  if (isLoading) return <Spin />
  if (error) return <Alert type="error" message={String(error)} />
  const study = data?.find((s) => s.id === selected) ?? data?.[0]
  if (!study) return <Panel><EmptyState title="没有跨框架实验归档" description="运行 benchmark.compare_frameworks_resnet 后可在此查看结果。" /></Panel>
  const rows = study.runs.filter((r) => r.mode === mode)
  const quality = study.runs.filter((r) => r.mode === 'quality' && r.curve.length)
  const p = study.protocol
  const metric = (title: string, key: keyof FrameworkRun, digits = 1, scale = 1) => ({
    title, align: 'right' as const,
    render: (_: unknown, row: FrameworkRun) => {
      const value = row[key]
      return typeof value === 'number' ? (value / scale).toFixed(digits) : '—'
    },
  })
  return <section className="page-section">
    <SectionHeader title="跨框架实验" description={`${p.architecture} · 训练 ${p.train} / 验证 ${p.val} · batch ${p.batch} · ${p.epochs} 轮`}
      extra={<Select aria-label="跨框架实验归档" value={study.id} onChange={setSelected}
        options={data?.map((s) => ({ value: s.id, label: s.id }))} />} />
    <Panel>
      <div className="toolbar tab-toolbar">
        <Select aria-label="测量口径" value={mode} onChange={setMode}
          options={Object.entries(MODE).map(([value, label]) => ({ value, label }))} />
      </div>
      <Table<FrameworkRun> rowKey={(r) => `${r.framework}-${r.seed}`} size="middle" pagination={false}
        scroll={{ x: 'max-content' }} dataSource={rows} columns={[
          { title: '框架', render: (_, r) => NAME[r.framework] ?? r.framework },
          { title: 'Seed', dataIndex: 'seed' }, metric('单步 ms', 'step_ms', 2), metric('samples/s', 'samples_per_s'),
          ...(mode === 'quality' ? [metric('验证转向 MAE', 'val_mae', 5)] : []),
          metric('GPU %', 'gpu_pct'), metric('整卡峰值 MiB', 'device_mib'), metric('CPU %', 'cpu_pct'),
          metric('RSS 峰值 MiB', 'rss_bytes', 1, 2**20),
          { title: '分配器峰值 MiB', align: 'right', render: (_, r) => <span title={r.allocator_kind ?? ''}>{r.allocator_mib?.toFixed(1) ?? '—'}</span> },
        ]} />
    </Panel>
    {quality.length > 0 && <ChartPanel title="验证转向 MAE" subtitle="相同划分与训练预算，逐种子显示" height={330} option={{
      animation: false, tooltip: modernTooltip, legend: { ...baseLegend, type: 'scroll' }, grid: baseGrid,
      xAxis: { type: 'value', name: 'Epoch', minInterval: 1 }, yAxis: { type: 'value', scale: true },
      series: quality.map((r) => ({ name: `${NAME[r.framework] ?? r.framework} · s${r.seed}`, type: 'line', data: r.curve })),
    }} />}
  </section>
}
