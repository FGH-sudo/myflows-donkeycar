import { LineChartOutlined, SearchOutlined } from '@ant-design/icons'
import { useQueries, useQuery } from '@tanstack/react-query'
import { Segmented, Select, Table } from 'antd'
import type { EChartsOption, SeriesOption } from 'echarts'
import { useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { api } from '../api'
import type { RunDetail, RunSummary } from '../api'
import ChartPanel from '../components/ChartPanel'
import EmptyState from '../components/EmptyState'
import { Panel, SectionHeader } from '../components/Panel'
import StatusTag from '../components/StatusTag'
import { RANK_COLORS, fmtDuration, fmtNum, modeLabel } from '../format'
import { baseLegend, modernTooltip, xAxisName } from './run/chartUtils'

const METRICS: { key: string; label: string; single?: string }[] = [
  { key: 'val_loss', label: 'Val Loss', single: 'val_loss' },
  { key: 'val_accuracy', label: 'Val Acc' },
  { key: 'val_angle_mae', label: '转向 MAE' },
  { key: 'samples_per_s', label: '吞吐' },
  { key: 'epoch_train_wall_s', label: 'Epoch 耗时', single: 'epoch_time_s' },
  { key: 'gradient_sync_s', label: '同步耗时' },
  { key: 'loss', label: 'Train Loss', single: 'loss' },
]

const UNITS: Record<string, string> = { samples_per_s: 'samples/s', epoch_train_wall_s: 's', gradient_sync_s: 's' }

const SYMBOLS = ['circle', 'rect', 'triangle', 'diamond', 'roundRect', 'pin']
const LINE_TYPES: ('solid' | 'dashed' | 'dotted')[] = ['solid', 'dashed', 'dotted']

function epochSeries(detail: RunDetail, metric: (typeof METRICS)[number]): [number, number | null][] {
  const key = detail.kind === 'distributed' ? metric.key : metric.single
  if (!key) return []
  const offset = detail.kind === 'distributed' ? 1 : 0
  return (detail.epochs ?? []).map((r) => [
    Number(r.epoch) + offset,
    typeof r[key] === 'number' ? (r[key] as number) : null,
  ])
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

  const option = useMemo<EChartsOption>(
    () => ({
      animation: false,
      tooltip: modernTooltip,
      legend: { ...baseLegend, type: 'scroll', right: 4 },
      grid: { left: 64, right: 32, top: 44, bottom: 48 },
      xAxis: { type: 'value', ...xAxisName('Epoch'), minInterval: 1 },
      yAxis: { type: 'value', scale: true },
      series: loaded.map((d, i): SeriesOption => {
        const color = RANK_COLORS[i % RANK_COLORS.length]
        return {
          name: d.summary.label,
          type: 'line',
          data: epochSeries(d, metricDef),
          symbol: SYMBOLS[i % SYMBOLS.length],
          symbolSize: 6,
          lineStyle: { color, width: 1.8, type: LINE_TYPES[i % LINE_TYPES.length] },
          itemStyle: { color },
        }
      }),
    }),
    [loaded, metricDef],
  )

  const options = (all.data ?? []).map((r: RunSummary) => ({
    value: r.id,
    label: `${r.label} · ${r.source === 'console' ? '控制台' : r.source}`,
    title: r.id,
  }))
  const rows = loaded.map((d) => d.summary)
  const baseline = rows.find((r) => r.mode === 'single')?.final?.epoch_train_wall_s

  return (
    <div>
      <div className="page-title">
        <div>
          <h2>运行对比</h2>
          <div className="page-subtitle">收敛曲线、训练速度与相对单进程的加速比</div>
        </div>
      </div>

      <div className="compare-picker">
        <Select
          mode="multiple"
          allowClear
          size="large"
          suffixIcon={<SearchOutlined />}
          style={{ width: '100%' }}
          placeholder="选择要对比的运行（也可在“运行记录”中勾选后跳转）"
          value={ids}
          options={options}
          showSearch
          optionFilterProp="label"
          loading={all.isLoading}
          onChange={(v) => setParams({ ids: (v as string[]).join(',') })}
          maxTagCount="responsive"
        />
      </div>

      {!ids.length ? (
        <Panel>
          <EmptyState icon={<LineChartOutlined />} title="至少选择 2 个运行" description="对比会叠加逐 Epoch 曲线，并列出最终指标与相对单进程的加速比" />
        </Panel>
      ) : (
        <>
          <section className="page-section" style={{ marginTop: 0 }}>
            <SectionHeader
              title="逐 Epoch 趋势"
              description={UNITS[metricDef.key] ? `${metricDef.label}，单位 ${UNITS[metricDef.key]}` : metricDef.label}
              extra={
                <Segmented
                  size="small"
                  value={metricDef.key}
                  onChange={(v) => setMetric(String(v))}
                  options={available.map((m) => ({ value: m.key, label: m.label }))}
                />
              }
            />
            <ChartPanel option={option} height={400} />
          </section>

          <section className="page-section">
            <SectionHeader title="最终结果" description={baseline ? '加速比以所选的单进程运行为基线' : '选择一个单进程运行即可计算加速比'} />
            <Panel>
              <Table<RunSummary>
                size="middle"
                rowKey="id"
                pagination={false}
                dataSource={rows}
                scroll={{ x: 'max-content' }}
                columns={[
                  {
                    title: '运行',
                    dataIndex: 'label',
                    render: (v, r) => (
                      <span className="nowrap" style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
                        <span className="status-dot" style={{ background: RANK_COLORS[rows.indexOf(r) % RANK_COLORS.length] }} />
                        <Link to={`/run/${r.id}`} className="table-link">
                          {v}
                        </Link>
                      </span>
                    ),
                  },
                  { title: '架构', render: (_, r) => <span className="nowrap" style={{ color: 'var(--text-secondary)' }}>{modeLabel(r)}</span> },
                  {
                    title: 'Worker',
                    dataIndex: 'workers',
                    align: 'center',
                    render: (v) => <span className="count-chip">{v}</span>,
                  },
                  { title: '状态', dataIndex: 'status', render: (s) => <StatusTag status={s} /> },
                  {
                    title: 'Val Loss',
                    align: 'right',
                    render: (_, r) => (
                      <span className="num" style={{ fontWeight: 600 }}>{fmtNum(r.final?.val?.loss)}</span>
                    ),
                  },
                  {
                    title: 'Val Acc',
                    align: 'right',
                    render: (_, r) => (
                      <span className="num">{fmtNum(r.final?.val?.accuracy)}</span>
                    ),
                  },
                  {
                    title: 'Test Acc',
                    align: 'right',
                    render: (_, r) => (
                      <span className="num">{fmtNum(r.final?.test?.accuracy)}</span>
                    ),
                  },
                  ...(rows.some(
                    (r) => r.final?.val?.angle_mae !== undefined && r.final?.val?.angle_mae !== null,
                  )
                    ? [
                        {
                          title: 'Val Angle MAE',
                          align: 'right' as const,
                          render: (_: unknown, r: RunSummary) => (
                            <span className="num">{fmtNum(r.final?.val?.angle_mae)}</span>
                          ),
                        },
                      ]
                    : []),
                  {
                    title: '末 Epoch 耗时',
                    align: 'right',
                    render: (_, r) => <span className="num">{fmtDuration(r.final?.epoch_train_wall_s)}</span>,
                  },
                  {
                    title: '加速比',
                    align: 'right',
                    render: (_, r) =>
                      baseline && r.final?.epoch_train_wall_s ? (
                        <span className="num" style={{ fontWeight: 600 }}>
                          {(baseline / r.final.epoch_train_wall_s).toFixed(2)}×
                        </span>
                      ) : (
                        <span className="muted">—</span>
                      ),
                  },
                  {
                    title: '吞吐',
                    align: 'right',
                    render: (_, r) =>
                      r.final?.samples_per_s ? (
                        <span className="num">
                          <span style={{ fontWeight: 600 }}>{fmtNum(r.final.samples_per_s, 0)}</span>
                          <span className="muted"> /s</span>
                        </span>
                      ) : (
                        <span className="muted">—</span>
                      ),
                  },
                  { title: '总耗时', align: 'right', render: (_, r) => <span className="num">{fmtDuration(r.elapsed_s)}</span> },
                ]}
              />
            </Panel>
          </section>
        </>
      )}
    </div>
  )
}
