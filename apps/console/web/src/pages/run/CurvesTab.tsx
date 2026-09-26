import { Checkbox, Col, Row, Segmented, Slider, Switch } from 'antd'
import type { EChartsOption, SeriesOption } from 'echarts'
import { useMemo, useState } from 'react'
import type { RunDetail } from '../../api'
import ChartPanel from '../../components/ChartPanel'
import EmptyState from '../../components/EmptyState'
import { Panel } from '../../components/Panel'
import { rankColor } from '../../format'
import type { RunState } from './useRunData'
import { baseGrid, baseLegend, ema, modernTooltip, plainGrid, xAxisName, zip, zoom } from './chartUtils'

interface Props {
  detail: RunDetail
  state: RunState
  ranks: number[]
}

const VAL_FIELDS: [string, string, string][] = [
  ['val_loss', 'val loss', '#3b6fd8'],
  ['val_accuracy', 'val accuracy', '#2f9e7a'],
  ['val_angle_mae', 'val angle MAE', '#d4892a'],
  ['val_mse', 'val MSE', '#8866d6'],
]

export function RankPicker({ ranks, value, onChange }: { ranks: number[]; value: number[]; onChange: (v: number[]) => void }) {
  if (ranks.length <= 1) return null
  return (
    <Checkbox.Group value={value} onChange={(v) => onChange(v as number[])}>
      {ranks.map((r) => (
        <Checkbox key={r} value={r}>
          <span className="toolbar-item" style={{ gap: 6, color: 'var(--text-main)' }}>
            <span className="stat-swatch" style={{ background: rankColor(r) }} />
            rank {r}
          </span>
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
  const xName = xMode === 'step' ? (distributed ? '迭代' : 'step') : '秒'

  const lossOption = useMemo<EChartsOption>(() => {
    const series: SeriesOption[] = []
    for (const rank of selected) {
      const cols = state.steps[String(rank)]
      if (!cols) continue
      const xs = xMode === 'step' ? cols.x : cols.t
      const color = rankColor(rank)
      const raw = cols.loss ?? []
      series.push({
        name: smooth > 0 ? `rank ${rank} 原始` : `rank ${rank}`,
        type: 'line',
        showSymbol: false,
        sampling: 'lttb',
        data: zip(xs, raw),
        lineStyle: { width: 1, opacity: smooth > 0 ? 0.22 : 1, color },
        itemStyle: { color, opacity: smooth > 0 ? 0.3 : 1 },
        z: 1,
      })
      if (smooth > 0) {
        series.push({
          name: `rank ${rank}`,
          type: 'line',
          showSymbol: false,
          sampling: 'lttb',
          data: zip(xs, ema(raw, smooth)),
          lineStyle: { width: 1.8, color },
          itemStyle: { color },
          z: 2,
        })
      }
    }
    return {
      animation: false,
      tooltip: { ...modernTooltip, valueFormatter: (v) => (typeof v === 'number' ? v.toPrecision(5) : String(v)) },
      legend: { ...baseLegend, type: 'scroll', right: 4 },
      grid: baseGrid,
      xAxis: { type: 'value', ...xAxisName(xName), scale: true },
      yAxis: { type: logY ? 'log' : 'value', scale: true },
      dataZoom: zoom,
      series,
    }
  }, [state.steps, selected, xMode, xName, smooth, logY])

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
      series.push({
        name: 'train loss',
        type: 'line',
        data: rows.map((r) => r.loss as number),
        lineStyle: { color: '#7a8394' },
        itemStyle: { color: '#7a8394' },
      })
    }
    if (distributed) {
      const perRank = allRanks.map((rank) => state.epochs[String(rank)] ?? [])
      if (perRank.some((r) => r.length)) {
        series.push({
          name: 'samples/s',
          type: 'bar',
          yAxisIndex: 1,
          barMaxWidth: 40,
          itemStyle: { color: '#ececf0', borderRadius: [4, 4, 0, 0] },
          data: rows.map((_, i) => {
            const items = perRank.map((r) => r[i]).filter(Boolean)
            if (!items.length) return null
            const start = Math.min(...items.map((r) => Number(r!.start_s)))
            const end = Math.max(...items.map((r) => Number(r!.end_s)))
            const samples = items.reduce((s, r) => s + Number(r!.samples ?? 0), 0)
            return end > start ? samples / (end - start) : null
          }),
        })
      }
    }
    for (const [field, label, color] of VAL_FIELDS) {
      if (rows.some((r) => typeof r[field] === 'number')) {
        series.push({
          name: label,
          type: 'line',
          data: rows.map((r) => (r[field] as number) ?? null),
          yAxisIndex: field === 'val_accuracy' ? 2 : 0,
          lineStyle: { color },
          itemStyle: { color },
        })
      }
    }
    const hasAcc = series.some((s) => s.name === 'val accuracy')
    return {
      animation: false,
      tooltip: modernTooltip,
      legend: { ...baseLegend, right: 4 },
      grid: { ...plainGrid, top: 60, right: hasAcc ? 110 : 60 },
      xAxis: { type: 'category', data: x, ...xAxisName('epoch') },
      yAxis: [
        { type: 'value', scale: true, name: 'loss' },
        { type: 'value', name: 'samples/s', splitLine: { show: false }, show: distributed },
        {
          type: 'value',
          name: 'acc',
          min: (v: { min: number }) => Math.max(0, Math.floor(v.min * 100 - 1) / 100),
          max: 1,
          offset: 56,
          splitLine: { show: false },
          show: hasAcc,
        },
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
      series.push({
        name: `rank ${rank}`,
        type: 'line',
        showSymbol: false,
        sampling: 'lttb',
        data: zip(xs, ema(values, Math.max(smooth, 0.3))),
        lineStyle: { width: 1.8, color: rankColor(rank) },
        itemStyle: { color: rankColor(rank) },
      })
    }
    return {
      animation: false,
      tooltip: { ...modernTooltip, valueFormatter: (v) => (typeof v === 'number' ? `${v.toFixed(0)} samples/s` : String(v)) },
      legend: baseLegend,
      grid: baseGrid,
      xAxis: { type: 'value', scale: true, ...xAxisName(xName) },
      yAxis: { type: 'value', scale: true },
      dataZoom: zoom,
      series,
    }
  }, [state.steps, selected, xMode, xName, smooth, distributed])

  const empty = !Object.values(state.steps).some((c) => (c.x?.length ?? 0) > 0)
  return (
    <div>
      <div className="toolbar tab-toolbar">
        <RankPicker ranks={allRanks} value={selected} onChange={setPicked} />
        <span className="toolbar-item">
          横轴
          <Segmented
            size="small"
            value={xMode}
            onChange={(v) => setXMode(v as 'step' | 'time')}
            options={[
              { value: 'step', label: distributed ? '迭代' : 'Step' },
              { value: 'time', label: '时间' },
            ]}
          />
        </span>
        <span className="toolbar-item">
          平滑
          <Slider style={{ width: 140, margin: '0 4px' }} min={0} max={0.98} step={0.02} value={smooth} onChange={setSmooth} />
          <span className="num" style={{ width: 32, color: 'var(--text-main)' }}>
            {smooth.toFixed(2)}
          </span>
        </span>
        <span className="toolbar-item">
          对数纵轴
          <Switch size="small" checked={logY} onChange={setLogY} />
        </span>
      </div>
      {empty ? (
        <Panel>
          <EmptyState title={state.loaded ? '还没有 step 数据' : '加载中…'} description="训练开始写入指标后，曲线会自动出现" />
        </Panel>
      ) : (
        <Row gutter={[16, 16]}>
          <Col span={24}>
            <ChartPanel title="训练 Loss" subtitle={smooth > 0 ? `EMA 平滑 ${smooth.toFixed(2)}，浅色为原始值` : '原始值'} option={lossOption} height={380} />
          </Col>
          {epochOption ? (
            <Col xs={24} xl={12}>
              <ChartPanel title="逐 Epoch 指标" subtitle={distributed ? '柱：全体 Worker 吞吐' : undefined} option={epochOption} height={340} />
            </Col>
          ) : null}
          <Col xs={24} xl={epochOption ? 12 : 24}>
            <ChartPanel
              title={distributed ? '单 Rank 吞吐' : '整步吞吐'}
              subtitle={distributed ? '本地样本数 / step 耗时，samples/s' : 'samples/s'}
              option={throughputOption}
              height={340}
            />
          </Col>
        </Row>
      )}
    </div>
  )
}
