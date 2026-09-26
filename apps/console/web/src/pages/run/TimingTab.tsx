import { Alert, Col, Row, Segmented } from 'antd'
import type { EChartsOption, SeriesOption } from 'echarts'
import { useMemo, useState } from 'react'
import type { RunDetail } from '../../api'
import ChartPanel from '../../components/ChartPanel'
import EmptyState from '../../components/EmptyState'
import { Panel } from '../../components/Panel'
import { fmtBytes, rankColor } from '../../format'
import type { RunState } from './useRunData'
import { baseGrid, baseLegend, binColumns, ema, modernTooltip, xAxisName, zip, zoom } from './chartUtils'

// Overheads are greys; compute stages carry the data palette.
const STAGES: [string, string, string][] = [
  ['data_s', '数据准备', '#b4b4bc'],
  ['input_h2d_wall_s', 'H2D 传输', '#d4d4d8'],
  ['forward_wall_s', '前向', '#3b6fd8'],
  ['backward_wall_s', '反向', '#2f9e7a'],
  ['optimizer_wall_s', '优化器', '#8866d6'],
  ['gradient_sync_s', '梯度同步', '#d4892a'],
]
const SINGLE_STAGES: [string, string, string][] = [
  ['data_load_ms', '数据加载', '#7a8394'],
  ['train_step_ms', '前向+反向', '#3b6fd8'],
  ['step_time_ms', '整步时间', '#d4892a'],
]
const OTHER_COLOR = '#ececf0'
const msLabel = { formatter: (v: number) => `${v} ms` }

export default function TimingTab({ detail, state, ranks }: { detail: RunDetail; state: RunState; ranks: number[] }) {
  const allRanks = ranks.length ? ranks : Object.keys(state.steps).map(Number)
  const [rank, setRank] = useState<number>(allRanks[0] ?? 0)
  const distributed = detail.kind === 'distributed'
  const cols = state.steps[String(rank)]

  const stackOption = useMemo<EChartsOption | null>(() => {
    if (!cols?.x?.length) return null
    const stages = distributed ? STAGES : SINGLE_STAGES
    const fields = stages.map((s) => s[0])
    const binned = binColumns(cols, fields, 'x', 400)
    const scale = distributed ? 1000 : 1
    const series: SeriesOption[] = stages
      .filter(([f]) => binned[f]?.some((v) => v !== null))
      .map(([field, label, color]) => ({
        name: label,
        type: 'line',
        showSymbol: false,
        stack: distributed ? 'stage' : undefined,
        areaStyle: distributed ? { opacity: 0.6 } : undefined,
        lineStyle: { width: distributed ? 0 : 1.6, color },
        itemStyle: { color },
        data: zip(
          binned.x,
          binned[field].map((v) => (v === null ? null : v * scale)),
        ),
      }))
    return {
      animation: false,
      tooltip: {
        ...modernTooltip,
        valueFormatter: (v) => (typeof v === 'number' ? `${v.toFixed(2)} ms` : String(v)),
      },
      legend: baseLegend,
      grid: { ...baseGrid, left: 64 },
      xAxis: { type: 'value', scale: true, ...xAxisName('迭代') },
      yAxis: { type: 'value', axisLabel: msLabel },
      dataZoom: zoom,
      series,
    }
  }, [cols, distributed])

  const shareOption = useMemo<EChartsOption | null>(() => {
    if (!distributed) return null
    const categories = allRanks.map((r) => `rank ${r}`)
    const totals = allRanks.map((r) => {
      const c = state.steps[String(r)]
      const out: Record<string, number> = {}
      for (const [field] of STAGES) out[field] = (c?.[field] ?? []).reduce<number>((s, v) => s + (v ?? 0), 0)
      out.step = (c?.step_wall_s ?? []).reduce<number>((s, v) => s + (v ?? 0), 0)
      return out
    })
    if (!totals.some((t) => t.step > 0)) return null
    const pct = (t: Record<string, number>, field: string) => (t.step ? (t[field] / t.step) * 100 : 0)
    const series: SeriesOption[] = STAGES.map(([field, label, color]) => ({
      name: label,
      type: 'bar',
      stack: 'share',
      barMaxWidth: 28,
      itemStyle: { color },
      data: totals.map((t) => pct(t, field)),
    }))
    series.push({
      name: '其他',
      type: 'bar',
      stack: 'share',
      itemStyle: { color: OTHER_COLOR },
      data: totals.map((t) => Math.max(0, 100 - STAGES.reduce((s, [f]) => s + pct(t, f), 0))),
    })
    return {
      animation: false,
      tooltip: {
        ...modernTooltip,
        axisPointer: { type: 'shadow' },
        valueFormatter: (v) => (typeof v === 'number' ? `${v.toFixed(1)}%` : String(v)),
      },
      legend: baseLegend,
      grid: { left: 64, right: 32, top: 64, bottom: 32 },
      xAxis: { type: 'value', max: 100, axisLabel: { formatter: '{value}%' } },
      yAxis: { type: 'category', data: categories, axisLine: { show: false } },
      series,
    }
  }, [state.steps, allRanks, distributed])

  const bytesOption = useMemo<EChartsOption | null>(() => {
    if (!distributed) return null
    const series: SeriesOption[] = allRanks.map((r) => {
      const c = state.steps[String(r)]
      const total = (c?.request_bytes ?? []).map((v, i) => (v ?? 0) + (c?.response_bytes?.[i] ?? 0))
      return {
        name: `rank ${r}`,
        type: 'line',
        showSymbol: false,
        sampling: 'lttb',
        data: zip(c?.x, ema(total, 0.5)),
        lineStyle: { color: rankColor(r), width: 1.8 },
        itemStyle: { color: rankColor(r) },
      }
    })
    return {
      animation: false,
      tooltip: {
        ...modernTooltip,
        valueFormatter: (v) => (typeof v === 'number' ? fmtBytes(v) : String(v)),
      },
      legend: baseLegend,
      grid: { ...baseGrid, left: 72 },
      xAxis: { type: 'value', scale: true, ...xAxisName('迭代') },
      yAxis: { type: 'value', axisLabel: { formatter: (v: number) => fmtBytes(v) } },
      dataZoom: zoom,
      series,
    }
  }, [state.steps, allRanks, distributed])

  if (!cols?.x?.length) {
    return (
      <Panel>
        <EmptyState title="还没有 step 计时数据" description="训练开始写入计时字段后自动显示" />
      </Panel>
    )
  }

  const pairHeight = Math.max(260, 150 + allRanks.length * 48)
  return (
    <div>
      {distributed && allRanks.length > 1 ? (
        <div className="toolbar tab-toolbar">
          <span className="toolbar-item">
            查看 Rank
            <Segmented
              size="small"
              value={rank}
              onChange={(v) => setRank(Number(v))}
              options={allRanks.map((r) => ({ value: r, label: `rank ${r}` }))}
            />
          </span>
        </div>
      ) : null}
      {distributed && detail.config?.profile === false ? (
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
          message="该运行关闭了 profile，只有 wall time，没有 CUDA Event 的 *_gpu_s 精确计时字段。"
        />
      ) : null}
      <Row gutter={[16, 16]}>
        {stackOption ? (
          <Col span={24}>
            <ChartPanel
              title={distributed ? `Rank ${rank} 每步阶段耗时` : '每步耗时'}
              subtitle={distributed ? '各阶段堆叠，单位 ms' : '单位 ms'}
              option={stackOption}
              height={380}
            />
          </Col>
        ) : null}
        {shareOption ? (
          <Col xs={24} xl={12}>
            <ChartPanel title="各 Rank 阶段耗时占比" subtitle="已加载 step 的累计占比" option={shareOption} height={pairHeight} />
          </Col>
        ) : null}
        {bytesOption ? (
          <Col xs={24} xl={12}>
            <ChartPanel title="每步通信量" subtitle="请求 + 响应" option={bytesOption} height={pairHeight} />
          </Col>
        ) : null}
      </Row>
    </div>
  )
}
