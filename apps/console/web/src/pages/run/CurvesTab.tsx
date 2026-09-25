import { Card, Checkbox, Col, Empty, Row, Segmented, Slider, Space, Switch } from 'antd'
import type { EChartsOption, SeriesOption } from 'echarts'
import { useMemo, useState } from 'react'
import type { RunDetail } from '../../api'
import EChart from '../../components/EChart'
import { rankColor } from '../../format'
import type { RunState } from './useRunData'
import { baseGrid, ema, zip, zoom } from './chartUtils'

interface Props {
  detail: RunDetail
  state: RunState
  ranks: number[]
}

const VAL_FIELDS: [string, string][] = [
  ['val_loss', 'val loss'],
  ['val_accuracy', 'val accuracy'],
  ['val_angle_mae', 'val angle MAE'],
  ['val_mse', 'val MSE'],
]

export function RankPicker({ ranks, value, onChange }: { ranks: number[]; value: number[]; onChange: (v: number[]) => void }) {
  if (ranks.length <= 1) return null
  return (
    <Checkbox.Group value={value} onChange={(v) => onChange(v as number[])}>
      {ranks.map((r) => (
        <Checkbox key={r} value={r}>
          <span style={{ color: rankColor(r), fontWeight: 600 }}>rank {r}</span>
        </Checkbox>
      ))}
    </Checkbox.Group>
  )
}

export default function CurvesTab({ detail, state, ranks }: Props) {
  const allRanks = ranks.length ? ranks : Object.keys(state.steps).map(Number)
  const [picked, setPicked] = useState<number[] | null>(null)
  const selected = picked ?? allRanks
  const [xMode, setXMode] = useState<'step' | 'time'>('step')
  const [smooth, setSmooth] = useState(0.6)
  const [logY, setLogY] = useState(false)
  const distributed = detail.kind === 'distributed'

  const lossOption = useMemo<EChartsOption>(() => {
    const series: SeriesOption[] = []
    for (const rank of selected) {
      const cols = state.steps[String(rank)]
      if (!cols) continue
      const xs = xMode === 'step' ? cols.x : cols.t
      const color = rankColor(rank)
      const raw = cols.loss ?? []
      series.push({ name: `rank ${rank} 原始`, type: 'line', showSymbol: false, sampling: 'lttb', data: zip(xs, raw),
        lineStyle: { width: 1, opacity: smooth > 0 ? 0.25 : 1, color }, itemStyle: { color }, z: 1 })
      if (smooth > 0) {
        series.push({ name: `rank ${rank}`, type: 'line', showSymbol: false, sampling: 'lttb', data: zip(xs, ema(raw, smooth)),
          lineStyle: { width: 2, color }, itemStyle: { color }, z: 2 })
      }
    }
    return {
      animation: false,
      title: { text: '训练 loss', left: 8, top: 6, textStyle: { fontSize: 14 } },
      tooltip: { trigger: 'axis', valueFormatter: (v) => (typeof v === 'number' ? v.toPrecision(5) : String(v)) },
      legend: { top: 6, right: 12, type: 'scroll', width: '60%' },
      grid: baseGrid,
      xAxis: { type: 'value', name: xMode === 'step' ? (distributed ? '迭代' : 'step') : '秒', nameLocation: 'middle', nameGap: 26, scale: true },
      yAxis: { type: logY ? 'log' : 'value', scale: true },
      dataZoom: zoom,
      series,
    }
  }, [state.steps, selected, xMode, smooth, logY, distributed])

  const epochOption = useMemo<EChartsOption | null>(() => {
    let rows: Record<string, unknown>[] = []
    let offset = 0
    if (distributed) {
      rows = (state.epochs['0'] ?? []) as Record<string, unknown>[]
      offset = 1
      if (!rows.length) rows = detail.epochs as Record<string, unknown>[]
    } else if (detail.kind === 'single') {
      rows = (state.epochs['0'] ?? detail.epochs) as Record<string, unknown>[]
    } else {
      rows = detail.epochs as Record<string, unknown>[]
    }
    if (!rows.length) return null
    const x = rows.map((r) => Number(r.epoch) + offset)
    const series: SeriesOption[] = []
    if (!distributed && rows.some((r) => typeof r.loss === 'number')) {
      series.push({ name: 'train loss', type: 'line', data: rows.map((r) => r.loss as number) })
    }
    if (distributed) {
      const perRank = allRanks.map((rank) => (state.epochs[String(rank)] ?? []))
      if (perRank.some((r) => r.length)) {
        series.push({ name: 'samples/s', type: 'bar', yAxisIndex: 1, itemStyle: { color: 'rgba(22,104,220,0.25)' },
          data: rows.map((_, i) => {
            const items = perRank.map((r) => r[i]).filter(Boolean)
            if (!items.length) return null
            const start = Math.min(...items.map((r) => Number(r!.start_s)))
            const end = Math.max(...items.map((r) => Number(r!.end_s)))
            const samples = items.reduce((s, r) => s + Number(r!.samples ?? 0), 0)
            return end > start ? samples / (end - start) : null
          }) })
      }
    }
    for (const [field, label] of VAL_FIELDS) {
      if (rows.some((r) => typeof r[field] === 'number')) {
        series.push({ name: label, type: 'line', symbolSize: 6, data: rows.map((r) => (r[field] as number) ?? null),
          yAxisIndex: field === 'val_accuracy' ? 2 : 0 })
      }
    }
    const hasAcc = series.some((s) => s.name === 'val accuracy')
    return {
      animation: false,
      title: { text: '逐 epoch 指标', left: 8, top: 6, textStyle: { fontSize: 14 } },
      tooltip: { trigger: 'axis' },
      legend: { bottom: 0, left: 'center' },
      grid: { ...baseGrid, top: 64, bottom: 64, right: hasAcc ? 110 : 60 },
      xAxis: { type: 'category', data: x, name: 'epoch', nameLocation: 'middle', nameGap: 26 },
      yAxis: [
        { type: 'value', scale: true, name: 'loss' },
        { type: 'value', name: 'samples/s', splitLine: { show: false }, show: distributed },
        { type: 'value', name: 'acc', min: (v: { min: number }) => Math.max(0, Math.floor(v.min * 100 - 1) / 100), max: 1,
          offset: 56, splitLine: { show: false }, show: hasAcc },
      ],
      series,
    }
  }, [state.epochs, detail, distributed, allRanks])

  const throughputOption = useMemo<EChartsOption>(() => {
    const series: SeriesOption[] = []
    for (const rank of selected) {
      const cols = state.steps[String(rank)]
      if (!cols) continue
      const xs = xMode === 'step' ? cols.x : cols.t
      let values: (number | null)[]
      if (distributed) {
        values = (cols.n_samples ?? []).map((n, i) => {
          const wall = cols.step_wall_s?.[i]
          return n !== null && wall ? n / wall : null
        })
      } else {
        values = cols.samples_per_sec ?? []
      }
      series.push({ name: `rank ${rank}`, type: 'line', showSymbol: false, sampling: 'lttb', data: zip(xs, ema(values, Math.max(smooth, 0.3))),
        lineStyle: { width: 1.5, color: rankColor(rank) }, itemStyle: { color: rankColor(rank) } })
    }
    return {
      animation: false,
      title: { text: distributed ? '单 rank 吞吐（本地样本 / step 时间）' : '吞吐', left: 8, top: 6, textStyle: { fontSize: 14 } },
      tooltip: { trigger: 'axis', valueFormatter: (v) => (typeof v === 'number' ? `${v.toFixed(0)} samples/s` : String(v)) },
      legend: { top: 6, right: 12 },
      grid: baseGrid,
      xAxis: { type: 'value', scale: true, name: xMode === 'step' ? '迭代' : '秒', nameLocation: 'middle', nameGap: 26 },
      yAxis: { type: 'value', scale: true },
      dataZoom: zoom,
      series,
    }
  }, [state.steps, selected, xMode, smooth, distributed])

  const empty = !Object.values(state.steps).some((c) => (c.x?.length ?? 0) > 0)
  return (
    <Space orientation="vertical" size={16} style={{ width: '100%' }}>
      <Card size="small">
        <Space wrap size={24}>
          <RankPicker ranks={allRanks} value={selected} onChange={setPicked} />
          <span>横轴 <Segmented size="small" value={xMode} onChange={(v) => setXMode(v as 'step' | 'time')}
            options={[{ value: 'step', label: 'step' }, { value: 'time', label: '时间' }]} /></span>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>平滑
            <Slider style={{ width: 140 }} min={0} max={0.98} step={0.02} value={smooth} onChange={setSmooth} /></span>
          <span>对数纵轴 <Switch size="small" checked={logY} onChange={setLogY} /></span>
        </Space>
      </Card>
      {empty ? <Card><Empty description={state.loaded ? '还没有 step 数据' : '加载中…'} /></Card> : (
        <Row gutter={[16, 16]}>
          <Col span={24}><Card size="small"><EChart option={lossOption} height={360} /></Card></Col>
          {epochOption ? <Col xs={24} xl={12}><Card size="small"><EChart option={epochOption} height={320} /></Card></Col> : null}
          <Col xs={24} xl={epochOption ? 12 : 24}><Card size="small"><EChart option={throughputOption} height={320} /></Card></Col>
        </Row>
      )}
    </Space>
  )
}
