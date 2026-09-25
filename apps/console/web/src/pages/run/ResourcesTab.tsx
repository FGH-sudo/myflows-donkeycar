import { Alert, Card, Col, Empty, Row } from 'antd'
import type { EChartsOption, SeriesOption } from 'echarts'
import { useMemo } from 'react'
import type { RunDetail } from '../../api'
import EChart from '../../components/EChart'
import MetricCard from '../../components/MetricCard'
import { rankColor } from '../../format'
import type { RunState } from './useRunData'
import { baseGrid, epochMarkAreas, zip, zoom } from './chartUtils'

function stats(values: (number | null)[] | undefined) {
  const v = (values ?? []).filter((x): x is number => x !== null && Number.isFinite(x))
  if (!v.length) return { mean: null, max: null }
  return { mean: v.reduce((a, b) => a + b, 0) / v.length, max: Math.max(...v) }
}

export default function ResourcesTab({ detail, state }: { detail: RunDetail; state: RunState }) {
  const res = state.resources
  const distributed = detail.kind === 'distributed'
  const marks = useMemo(() => epochMarkAreas(state.epochs['0'], distributed ? 1 : 0), [state.epochs, distributed])

  const option = (title: string, series: SeriesOption[], yAxis: EChartsOption['yAxis']): EChartsOption => ({
    animation: false,
    title: { text: title, left: 8, top: 6, textStyle: { fontSize: 14 } },
    tooltip: { trigger: 'axis', valueFormatter: (v) => (typeof v === 'number' ? v.toFixed(1) : String(v)) },
    legend: { top: 6, right: 12 },
    grid: baseGrid,
    xAxis: { type: 'value', name: '秒（与 step 同一时钟）', nameLocation: 'middle', nameGap: 26, scale: true },
    yAxis,
    dataZoom: zoom,
    series,
  })

  const gpuOption = useMemo(() => {
    if (!res) return null
    const series: SeriesOption[] = [
      { name: 'GPU 利用率 %', type: 'line', showSymbol: false, areaStyle: { opacity: 0.12 }, data: zip(res.t, res.gpu_utilization_pct),
        markArea: marks.length ? { silent: true, label: { position: 'insideTop', color: '#1668dc', fontSize: 10 }, data: marks as never } : undefined },
      { name: 'CPU 利用率 %', type: 'line', showSymbol: false, lineStyle: { type: 'dashed' }, data: zip(res.t, res.cpu_utilization_pct) },
      { name: '显存 MiB', type: 'line', showSymbol: false, yAxisIndex: 1, data: zip(res.t, res.gpu_memory_mib) },
    ]
    return option('GPU / CPU 利用率与显存（整卡）', series, [
      { type: 'value', min: 0, max: 100, name: '%' },
      { type: 'value', name: 'MiB', splitLine: { show: false } },
    ])
  }, [res, marks])

  const powerOption = useMemo(() => {
    if (!res) return null
    return option('功耗与温度', [
      { name: '功耗 W', type: 'line', showSymbol: false, data: zip(res.t, res.gpu_power_w) },
      { name: '温度 °C', type: 'line', showSymbol: false, yAxisIndex: 1, data: zip(res.t, res.gpu_temperature_c) },
    ], [{ type: 'value', name: 'W' }, { type: 'value', name: '°C', splitLine: { show: false }, scale: true }])
  }, [res])

  const rssOption = useMemo(() => {
    if (!res || !Object.keys(res.rank_rss_mib ?? {}).length) return null
    const series: SeriesOption[] = Object.entries(res.rank_rss_mib).map(([rank, values]) => ({
      name: `rank ${rank}`, type: 'line', showSymbol: false, data: zip(res.t, values),
      lineStyle: { color: rankColor(rank) }, itemStyle: { color: rankColor(rank) },
    }))
    return option('各 rank 进程内存（RSS）', series, { type: 'value', name: 'MiB', scale: true })
  }, [res])

  if (!res || !res.t?.length) {
    return (
      <Card>
        <Empty description={detail.kind === 'tensorboard'
          ? '历史 TensorBoard 运行没有资源采样记录'
          : detail.config?.monitor === false ? '该运行关闭了资源监控（monitor=false）' : '还没有资源采样数据'} />
      </Card>
    )
  }
  const util = stats(res.gpu_utilization_pct)
  const mem = stats(res.gpu_memory_mib)
  const power = stats(res.gpu_power_w)
  const cpu = stats(res.cpu_utilization_pct)
  return (
    <Row gutter={[16, 16]}>
      <Col span={24}>
        <Alert type="info" showIcon message="GPU 指标是整卡读数：所有 Worker 共享同一张 GPU，不能区分单个 Worker 的利用率。阴影区为各 epoch 的训练区间。" />
      </Col>
      <Col xs={12} lg={6}><MetricCard label="GPU 利用率 均值 / 峰值" value={util.mean !== null ? util.mean.toFixed(0) : '—'} unit={util.max !== null ? `% / ${util.max.toFixed(0)}%` : ''} /></Col>
      <Col xs={12} lg={6}><MetricCard label="显存 峰值（硬件采样）" value={mem.max !== null ? mem.max.toFixed(0) : '—'} unit="MiB" /></Col>
      <Col xs={12} lg={6}><MetricCard label="功耗 均值 / 峰值" value={power.mean !== null ? power.mean.toFixed(1) : '—'} unit={power.max !== null ? `W / ${power.max.toFixed(1)} W` : ''} /></Col>
      <Col xs={12} lg={6}><MetricCard label="CPU 利用率 均值" value={cpu.mean !== null ? cpu.mean.toFixed(0) : '—'} unit={`% · ${res.t.length} 个采样点`} /></Col>
      {gpuOption ? <Col span={24}><Card size="small"><EChart option={gpuOption} height={340} /></Card></Col> : null}
      {powerOption ? <Col xs={24} xl={rssOption ? 12 : 24}><Card size="small"><EChart option={powerOption} height={300} /></Card></Col> : null}
      {rssOption ? <Col xs={24} xl={12}><Card size="small"><EChart option={rssOption} height={300} /></Card></Col> : null}
    </Row>
  )
}
