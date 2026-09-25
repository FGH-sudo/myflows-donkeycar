import { Alert, Card, Col, Empty, Row, Segmented, Space } from 'antd'
import type { EChartsOption, SeriesOption } from 'echarts'
import { useMemo, useState } from 'react'
import type { RunDetail } from '../../api'
import EChart from '../../components/EChart'
import { fmtBytes, rankColor } from '../../format'
import type { RunState } from './useRunData'
import { baseGrid, binColumns, ema, zip, zoom } from './chartUtils'

const STAGES: [string, string, string][] = [
  ['data_s', '数据', '#8c8c8c'],
  ['input_h2d_wall_s', 'H2D', '#bfbfbf'],
  ['forward_wall_s', 'forward', '#1677ff'],
  ['backward_wall_s', 'backward', '#13c2c2'],
  ['optimizer_wall_s', 'optimizer', '#52c41a'],
  ['gradient_sync_s', '梯度同步', '#fa541c'],
]
const SINGLE_STAGES: [string, string, string][] = [
  ['data_load_ms', '数据读取', '#8c8c8c'],
  ['train_step_ms', '前向+反向', '#1677ff'],
  ['step_time_ms', '整步', '#fa541c'],
]

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
        name: label, type: 'line', showSymbol: false, stack: distributed ? 'stage' : undefined,
        areaStyle: distributed ? { opacity: 0.75 } : undefined, lineStyle: { width: distributed ? 0.5 : 1.5, color }, itemStyle: { color },
        data: zip(binned.x, binned[field].map((v) => (v === null ? null : v * scale))),
      }))
    return {
      animation: false,
      title: { text: distributed ? `rank ${rank} 每步各阶段耗时（堆叠，ms）` : '每步耗时（ms）', left: 8, top: 6, textStyle: { fontSize: 14 } },
      tooltip: { trigger: 'axis', valueFormatter: (v) => (typeof v === 'number' ? `${v.toFixed(2)} ms` : String(v)) },
      legend: { top: 6, right: 12, type: 'scroll', width: '55%' },
      grid: baseGrid,
      xAxis: { type: 'value', scale: true, name: '迭代', nameLocation: 'middle', nameGap: 26 },
      yAxis: { type: 'value', name: 'ms' },
      dataZoom: zoom,
      series,
    }
  }, [cols, rank, distributed])

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
      name: label, type: 'bar', stack: 'share', itemStyle: { color },
      data: totals.map((t) => pct(t, field)),
    }))
    series.push({
      name: '其他', type: 'bar', stack: 'share', itemStyle: { color: '#f0f0f0' },
      data: totals.map((t) => Math.max(0, 100 - STAGES.reduce((s, [f]) => s + pct(t, f), 0))),
    })
    return {
      animation: false,
      title: { text: '各 rank 阶段耗时占比', subtext: '按已加载 step 求和；“其他”为整步中未归入任何阶段的时间', left: 8, top: 4, textStyle: { fontSize: 14 } },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: (v) => (typeof v === 'number' ? `${v.toFixed(1)}%` : String(v)) },
      legend: { bottom: 0, left: 'center' },
      grid: { left: 70, right: 24, top: 64, bottom: 56 },
      xAxis: { type: 'value', max: 100, axisLabel: { formatter: '{value}%' } },
      yAxis: { type: 'category', data: categories },
      series,
    }
  }, [state.steps, allRanks, distributed])

  const bytesOption = useMemo<EChartsOption | null>(() => {
    if (!distributed) return null
    const series: SeriesOption[] = allRanks.map((r) => {
      const c = state.steps[String(r)]
      const total = (c?.request_bytes ?? []).map((v, i) => (v ?? 0) + (c?.response_bytes?.[i] ?? 0))
      return { name: `rank ${r}`, type: 'line', showSymbol: false, sampling: 'lttb', data: zip(c?.x, ema(total, 0.5)),
        lineStyle: { color: rankColor(r) }, itemStyle: { color: rankColor(r) } }
    })
    return {
      animation: false,
      title: { text: '每步通信字节（请求 + 响应）', left: 8, top: 6, textStyle: { fontSize: 14 } },
      tooltip: { trigger: 'axis', valueFormatter: (v) => (typeof v === 'number' ? fmtBytes(v) : String(v)) },
      legend: { top: 6, right: 12 },
      grid: baseGrid,
      xAxis: { type: 'value', scale: true, name: '迭代', nameLocation: 'middle', nameGap: 26 },
      yAxis: { type: 'value', axisLabel: { formatter: (v: number) => fmtBytes(v) } },
      dataZoom: zoom,
      series,
    }
  }, [state.steps, allRanks, distributed])

  if (!cols?.x?.length) return <Card><Empty description="还没有 step 计时数据" /></Card>
  return (
    <Space orientation="vertical" size={16} style={{ width: '100%' }}>
      {distributed && allRanks.length > 1 ? (
        <Card size="small">
          选择 rank <Segmented value={rank} onChange={(v) => setRank(Number(v))} options={allRanks.map((r) => ({ value: r, label: `rank ${r}` }))} />
        </Card>
      ) : null}
      {distributed && detail.config?.profile === false ? (
        <Alert type="info" showIcon message="该运行关闭了 profile，只有 wall time，没有 CUDA Event 的 *_gpu_s 字段。" />
      ) : null}
      <Row gutter={[16, 16]}>
        {stackOption ? <Col span={24}><Card size="small"><EChart option={stackOption} height={340} /></Card></Col> : null}
        {shareOption ? <Col xs={24} xl={12}><Card size="small"><EChart option={shareOption} height={130 + allRanks.length * 44} /></Card></Col> : null}
        {bytesOption ? <Col xs={24} xl={12}><Card size="small"><EChart option={bytesOption} height={300} /></Card></Col> : null}
      </Row>
    </Space>
  )
}
