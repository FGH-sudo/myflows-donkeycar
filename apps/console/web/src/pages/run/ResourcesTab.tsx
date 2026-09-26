import { Col, Row } from 'antd'
import type { EChartsOption, SeriesOption } from 'echarts'
import * as echarts from 'echarts'
import { useMemo } from 'react'
import type { RunDetail } from '../../api'
import ChartPanel from '../../components/ChartPanel'
import EmptyState from '../../components/EmptyState'
import { Panel } from '../../components/Panel'
import StatStrip from '../../components/StatStrip'
import { rankColor } from '../../format'
import type { RunState } from './useRunData'
import { baseGrid, baseLegend, epochMarkAreas, modernTooltip, xAxisName, zip, zoom } from './chartUtils'

function stats(values: (number | null)[] | undefined) {
  const v = (values ?? []).filter((x): x is number => x !== null && Number.isFinite(x))
  if (!v.length) return { mean: null, max: null }
  return { mean: v.reduce((a, b) => a + b, 0) / v.length, max: Math.max(...v) }
}

export default function ResourcesTab({ detail, state }: { detail: RunDetail; state: RunState }) {
  const res = state.resources
  const distributed = detail.kind === 'distributed'
  const marks = useMemo(() => epochMarkAreas(state.epochs['0'], distributed ? 1 : 0), [state.epochs, distributed])

  const option = (series: SeriesOption[], yAxis: EChartsOption['yAxis']): EChartsOption => ({
    animation: false,
    tooltip: { ...modernTooltip, valueFormatter: (v) => (typeof v === 'number' ? v.toFixed(1) : String(v)) },
    legend: baseLegend,
    grid: { ...baseGrid, left: 64, right: 72 },
    xAxis: { type: 'value', ...xAxisName('秒（统一系统时钟）'), scale: true },
    yAxis,
    dataZoom: zoom,
    series,
  })

  const gpuOption = useMemo(() => {
    if (!res) return null
    const series: SeriesOption[] = [
      {
        name: 'GPU 利用率',
        type: 'line',
        showSymbol: false,
        lineStyle: { width: 1.6, color: '#3b6fd8' },
        itemStyle: { color: '#3b6fd8' },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(59, 111, 216, 0.14)' },
            { offset: 1, color: 'rgba(59, 111, 216, 0)' },
          ]),
        },
        data: zip(res.t, res.gpu_utilization_pct),
        markArea: marks.length
          ? {
              silent: true,
              label: {
                position: 'insideBottom',
                distance: 6,
                color: '#52525b',
                fontSize: 10,
                fontWeight: 600,
                backgroundColor: 'rgba(255, 255, 255, 0.85)',
                padding: [2, 5],
                borderRadius: 4,
              },
              data: marks as never,
            }
          : undefined,
      },
      {
        name: 'CPU 利用率',
        type: 'line',
        showSymbol: false,
        lineStyle: { type: 'dashed', width: 1.2, color: '#7a8394' },
        itemStyle: { color: '#7a8394' },
        data: zip(res.t, res.cpu_utilization_pct),
      },
      {
        name: '显存',
        type: 'line',
        showSymbol: false,
        yAxisIndex: 1,
        lineStyle: { width: 1.6, color: '#8866d6' },
        itemStyle: { color: '#8866d6' },
        data: zip(res.t, res.gpu_memory_mib),
      },
    ]
    return option(series, [
      { type: 'value', min: 0, max: 100, axisLabel: { formatter: '{value}%' } },
      { type: 'value', axisLabel: { formatter: (v: number) => (v >= 1024 ? `${(v / 1024).toFixed(1)} GB` : `${v} MB`) }, splitLine: { show: false } },
    ])
  }, [res, marks])

  const powerOption = useMemo(() => {
    if (!res) return null
    return option(
      [
        {
          name: '功耗',
          type: 'line',
          showSymbol: false,
          lineStyle: { width: 1.6, color: '#d4892a' },
          itemStyle: { color: '#d4892a' },
          data: zip(res.t, res.gpu_power_w),
        },
        {
          name: '温度',
          type: 'line',
          showSymbol: false,
          yAxisIndex: 1,
          lineStyle: { width: 1.6, color: '#cf5a80' },
          itemStyle: { color: '#cf5a80' },
          data: zip(res.t, res.gpu_temperature_c),
        },
      ],
      [
        { type: 'value', axisLabel: { formatter: '{value} W' } },
        { type: 'value', axisLabel: { formatter: '{value} °C' }, splitLine: { show: false }, scale: true },
      ],
    )
  }, [res])

  const rssOption = useMemo(() => {
    if (!res || !Object.keys(res.rank_rss_mib ?? {}).length) return null
    const series: SeriesOption[] = Object.entries(res.rank_rss_mib).map(([rank, values]) => ({
      name: `rank ${rank}`,
      type: 'line',
      showSymbol: false,
      data: zip(res.t, values),
      lineStyle: { color: rankColor(rank), width: 1.6 },
      itemStyle: { color: rankColor(rank) },
    }))
    return option(series, {
      type: 'value',
      scale: true,
      axisLabel: { formatter: (v: number) => `${v} MB` },
    })
  }, [res])

  if (!res || !res.t?.length) {
    return (
      <Panel>
        <EmptyState
          title="没有资源采样数据"
          description={
            detail.kind === 'tensorboard'
              ? '历史 TensorBoard 运行没有关联硬件资源采样记录'
              : detail.config?.monitor === false
                ? '该运行关闭了资源监控（monitor=false）'
                : '训练开始后会按固定间隔采样整卡与进程资源'
          }
        />
      </Panel>
    )
  }

  const util = stats(res.gpu_utilization_pct)
  const mem = stats(res.gpu_memory_mib)
  const power = stats(res.gpu_power_w)
  const cpu = stats(res.cpu_utilization_pct)

  return (
    <div>
      <div style={{ marginBottom: 16 }}>
        <StatStrip
          items={[
            {
              label: 'GPU 利用率',
              color: '#3b6fd8',
              value: util.mean !== null ? util.mean.toFixed(0) : '—',
              unit: '% 均值',
              hint: util.max !== null ? `峰值 ${util.max.toFixed(0)}%` : undefined,
              spark: res.gpu_utilization_pct,
              sparkRange: [0, 100],
            },
            {
              label: '显存峰值',
              color: '#8866d6',
              value: mem.max !== null ? (mem.max / 1024).toFixed(2) : '—',
              unit: 'GB',
              hint: mem.mean !== null ? `均值 ${(mem.mean / 1024).toFixed(2)} GB · 整卡` : undefined,
              spark: res.gpu_memory_mib,
            },
            {
              label: 'GPU 功耗',
              color: '#d4892a',
              value: power.mean !== null ? power.mean.toFixed(1) : '—',
              unit: 'W 均值',
              hint: power.max !== null ? `峰值 ${power.max.toFixed(1)} W` : undefined,
              spark: res.gpu_power_w,
            },
            {
              label: 'CPU 利用率',
              color: '#7a8394',
              value: cpu.mean !== null ? cpu.mean.toFixed(0) : '—',
              unit: '% 均值',
              hint: `${res.t.length} 个采样点`,
              spark: res.cpu_utilization_pct,
              sparkRange: [0, 100],
            },
          ]}
        />
      </div>

      <Row gutter={[16, 16]}>
        {gpuOption ? (
          <Col span={24}>
            <ChartPanel
              title="GPU / CPU 利用率与显存"
              subtitle="整卡读数，所有 Worker 共享同一张 GPU；浅色竖条为各 Epoch 的训练时段"
              option={gpuOption}
              height={360}
            />
          </Col>
        ) : null}

        {powerOption ? (
          <Col xs={24} xl={rssOption ? 12 : 24}>
            <ChartPanel title="功耗与温度" subtitle="左轴功耗，右轴温度" option={powerOption} height={320} />
          </Col>
        ) : null}

        {rssOption ? (
          <Col xs={24} xl={12}>
            <ChartPanel title="各 Rank 进程内存" subtitle="常驻内存 RSS" option={rssOption} height={320} />
          </Col>
        ) : null}
      </Row>
    </div>
  )
}
