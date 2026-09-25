import { useQueries, useQuery } from '@tanstack/react-query'
import { Card, Col, Empty, Row, Segmented, Select, Table } from 'antd'
import type { EChartsOption, SeriesOption } from 'echarts'
import { useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { api } from '../api'
import type { RunDetail, RunSummary } from '../api'
import EChart from '../components/EChart'
import StatusTag from '../components/StatusTag'
import { RANK_COLORS, fmtDuration, fmtNum, modeLabel } from '../format'

const METRICS: { key: string; label: string; single?: string }[] = [
  { key: 'val_loss', label: 'val loss', single: 'val_loss' },
  { key: 'val_accuracy', label: 'val accuracy' },
  { key: 'val_angle_mae', label: 'val angle MAE' },
  { key: 'samples_per_s', label: '吞吐 samples/s' },
  { key: 'epoch_train_wall_s', label: 'epoch 时间 s', single: 'epoch_time_s' },
  { key: 'gradient_sync_s', label: '梯度同步 s' },
  { key: 'loss', label: 'train loss（单进程）', single: 'loss' },
]

const SYMBOLS = ['circle', 'rect', 'triangle', 'diamond', 'roundRect', 'pin']
const LINE_TYPES: ('solid' | 'dashed' | 'dotted')[] = ['solid', 'dashed', 'dotted']

function epochSeries(detail: RunDetail, metric: (typeof METRICS)[number]): [number, number | null][] {
  const key = detail.kind === 'distributed' ? metric.key : metric.single
  if (!key) return []
  const offset = detail.kind === 'distributed' ? 1 : 0
  return (detail.epochs ?? []).map((r) => [Number(r.epoch) + offset, typeof r[key] === 'number' ? (r[key] as number) : null])
}

export default function Compare() {
  const [params, setParams] = useSearchParams()
  const ids = (params.get('ids') ?? '').split(',').filter(Boolean)
  const [metric, setMetric] = useState(METRICS[0].key)
  const all = useQuery({ queryKey: ['runs', 'all'], queryFn: () => api.runs() })
  const details = useQueries({ queries: ids.map((id) => ({ queryKey: ['run', id], queryFn: () => api.run(id) })) })
  const loaded = details.map((d) => d.data).filter((d): d is RunDetail => !!d)
  const available = METRICS.filter((m) => loaded.some((d) => epochSeries(d, m).some(([, v]) => v !== null)))
  const metricDef = available.find((m) => m.key === metric) ?? available[0] ?? METRICS[0]

  const option = useMemo<EChartsOption>(() => ({
    animation: false,
    tooltip: { trigger: 'axis' },
    legend: { top: 4, type: 'scroll' },
    grid: { left: 60, right: 30, top: 50, bottom: 40 },
    xAxis: { type: 'value', name: 'epoch', nameLocation: 'middle', nameGap: 26, minInterval: 1 },
    yAxis: { type: 'value', scale: true, name: metricDef.label },
    series: loaded.map((d, i): SeriesOption => {
      const color = RANK_COLORS[i % RANK_COLORS.length]
      return {
        name: d.summary.label, type: 'line', data: epochSeries(d, metricDef),
        symbol: SYMBOLS[i % SYMBOLS.length], symbolSize: 8,
        lineStyle: { color, width: 2, type: LINE_TYPES[i % LINE_TYPES.length] }, itemStyle: { color },
      }
    }),
  }), [loaded, metricDef])

  const options = (all.data ?? []).map((r: RunSummary) => ({ value: r.id, label: `${r.label} · ${r.source}` }))
  const rows = loaded.map((d) => d.summary)
  const baseline = rows.find((r) => r.mode === 'single')?.final?.epoch_train_wall_s

  return (
    <div>
      <div className="page-title"><h2>运行对比</h2></div>
      <Card size="small" style={{ marginBottom: 16 }}>
        <Select mode="multiple" style={{ width: '100%' }} placeholder="选择要对比的运行（也可在“运行记录”里勾选）" value={ids}
          options={options} showSearch optionFilterProp="label" loading={all.isLoading}
          onChange={(v) => setParams({ ids: (v as string[]).join(',') })} maxTagCount="responsive" />
      </Card>
      {!ids.length ? <Card><Empty description="至少选择两个运行" /></Card> : (
        <Row gutter={[16, 16]}>
          <Col span={24}>
            <Card size="small" title="逐 epoch 曲线" extra={<Segmented size="small" value={metricDef.key} onChange={(v) => setMetric(String(v))}
              options={available.map((m) => ({ value: m.key, label: m.label }))} />}>
              <EChart option={option} height={380} />
            </Card>
          </Col>
          <Col span={24}>
            <Card size="small" title="最终结果">
              <Table<RunSummary> size="small" rowKey="id" pagination={false} dataSource={rows} scroll={{ x: true }}
                columns={[
                  { title: '运行', dataIndex: 'label', render: (v, r) => <Link to={`/run/${r.id}`}>{v}</Link> },
                  { title: '模式', render: (_, r) => modeLabel(r) },
                  { title: 'Worker', dataIndex: 'workers' },
                  { title: '状态', dataIndex: 'status', render: (s) => <StatusTag status={s} /> },
                  { title: 'val loss', render: (_, r) => fmtNum(r.final?.val?.loss) },
                  { title: 'val acc', render: (_, r) => fmtNum(r.final?.val?.accuracy) },
                  { title: 'test acc', render: (_, r) => fmtNum(r.final?.test?.accuracy) },
                  ...(rows.some((r) => r.final?.val?.angle_mae !== undefined && r.final?.val?.angle_mae !== null)
                    ? [{ title: 'val angle MAE', render: (_: unknown, r: RunSummary) => fmtNum(r.final?.val?.angle_mae) }] : []),
                  { title: '末 epoch 时间', render: (_, r) => fmtDuration(r.final?.epoch_train_wall_s) },
                  { title: '相对单进程', render: (_, r) => (baseline && r.final?.epoch_train_wall_s ? `${(baseline / r.final.epoch_train_wall_s).toFixed(2)}×` : '—') },
                  { title: '吞吐', render: (_, r) => (r.final?.samples_per_s ? `${fmtNum(r.final.samples_per_s, 0)}/s` : '—') },
                  { title: '总耗时', render: (_, r) => fmtDuration(r.elapsed_s) },
                ]} />
            </Card>
          </Col>
        </Row>
      )}
    </div>
  )
}
