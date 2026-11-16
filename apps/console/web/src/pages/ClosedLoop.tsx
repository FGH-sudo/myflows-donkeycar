import { useQuery } from '@tanstack/react-query'
import { Alert, Segmented, Select, Spin, Table, Tag } from 'antd'
import type { EChartsOption, SeriesOption } from 'echarts'
import { useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api } from '../api'
import type { ClosedLoopDetail, ClosedLoopEpisode, ClosedLoopModel, ClosedLoopTrace } from '../api'
import ChartPanel from '../components/ChartPanel'
import EmptyState from '../components/EmptyState'
import { Panel, SectionHeader } from '../components/Panel'
import { RANK_COLORS, fmtNum, fmtTime } from '../format'
import { baseLegend, baseSplitLine, modernTooltip, plainGrid, xAxisName } from './run/chartUtils'

const METRICS: { key: string; label: string; unit?: string; digits: number }[] = [
  { key: 'mean_abs_cte', label: '平均 |CTE|', digits: 3 },
  { key: 'max_abs_cte', label: '最大 |CTE|', digits: 3 },
  { key: 'excursions', label: '出界次数', digits: 1 },
  { key: 'survival_s', label: '存活时间', unit: 's', digits: 2 },
  { key: 'distance_units', label: '行驶距离', digits: 1 },
  { key: 'inference_p50_ms', label: '推理 P50', unit: 'ms', digits: 2 },
  { key: 'inference_p95_ms', label: '推理 P95', unit: 'ms', digits: 2 },
]

/** Width over height of the top view's plot area: a 640 px panel body with 12 px side padding and a 420 px
 * chart, less the grid margins. */
const PATH_ASPECT = (640 - 24 - 56 - 88) / (420 - 24 - 44)

const TERMINATION: Record<string, string> = {
  time_limit: '到时', collision: '碰撞', excursion: '出界', error: '出错', stuck: '卡住', road_lost: '丢路',
}
/** Udacity archives estimate CTE from the camera and end episodes by their own failure rules, not collisions. */
const cameraCte = (detail: { cte_source?: string | null }) => detail.cte_source === 'camera_estimate'
const OUTSIDE = '#d64545'

const num = (value: unknown): number | null => (typeof value === 'number' && Number.isFinite(value) ? value : null)
/** A difference with its sign; a value that rounds to zero is shown without one. */
function signed(value: number | null | undefined, digits: number) {
  if (value === null || value === undefined) return '—'
  const text = Math.abs(value).toFixed(digits)
  if (Number(text) === 0) return text
  return `${value > 0 ? '+' : '−'}${text}`
}

/** Whether a difference from the baseline is an improvement, a regression or neither (zero / unknown). */
function direction(metric: string, value: number | null | undefined, higherIsBetter: string[]) {
  if (!value) return 'same'
  return value > 0 === higherIsBetter.includes(metric) ? 'better' : 'worse'
}

const DIRECTION_COLOR = { better: 'var(--success, #2f9e7a)', worse: OUTSIDE, same: 'var(--text-secondary)' }

/** Scenario ids in protocol order; an archive without a protocol falls back to the ids its episodes used. */
const scenarioIds = (detail: ClosedLoopDetail) =>
  detail.scenario_ids.length ? detail.scenario_ids : [...new Set(detail.episodes.map((e) => e.scenario))]

function ModelTable({ detail }: { detail: ClosedLoopDetail }) {
  const stat = (key: string, digits: number) => ({
    title: METRICS.find((m) => m.key === key)?.label ?? key,
    align: 'right' as const,
    render: (_: unknown, row: ClosedLoopModel) => {
      const mean = row.metrics[key]?.mean
      const worst = row.metrics[key]?.worst
      return (
        <span className="nowrap" title={worst === null || worst === undefined ? undefined : `最差场景 ${worst.toFixed(digits)}`}>
          {fmtNum(mean, digits)}
        </span>
      )
    },
  })
  return (
    <Table<ClosedLoopModel>
      rowKey="id"
      size="middle"
      pagination={false}
      scroll={{ x: 'max-content' }}
      dataSource={detail.models}
      columns={[
        {
          title: '模型',
          render: (_, row) => (
            <span className="nowrap" style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }} title={row.path ?? undefined}>
              <span className="status-dot" style={{ background: RANK_COLORS[detail.models.indexOf(row) % RANK_COLORS.length] }} />
              {row.id}
              {row.baseline ? <Tag>基线</Tag> : null}
            </span>
          ),
        },
        {
          title: '离线转向 MAE',
          align: 'right',
          render: (_, row) => fmtNum(row.offline_angle_mae, 4),
        },
        stat('mean_abs_cte', 3),
        stat('max_abs_cte', 3),
        stat('excursions', 1),
        stat('survival_s', 2),
        { ...stat('distance_units', 1), title: detail.distance_unit === 'm' ? '行驶距离 (m)' : '行驶距离' },
        {
          title: '结束方式',
          render: (_, row) => (
            <span className="nowrap" style={{ color: 'var(--text-secondary)' }}>
              到时 {row.time_limits ?? '—'} ·{' '}
              {cameraCte(detail) ? `失败 ${row.failures ?? '—'}` : `碰撞 ${row.collisions ?? '—'}`} / {row.episodes ?? '—'}
            </span>
          ),
        },
        {
          title: '推理 P50 / P95 ms',
          align: 'right',
          render: (_, row) => (
            <span className="nowrap">
              {fmtNum(row.metrics.inference_p50_ms?.mean, 1)} / {fmtNum(row.metrics.inference_p95_ms?.mean, 1)}
            </span>
          ),
        },
      ]}
    />
  )
}

function PairedSection({ detail }: { detail: ClosedLoopDetail }) {
  const [metric, setMetric] = useState(METRICS[0].key)
  const def = METRICS.find((m) => m.key === metric) ?? METRICS[0]
  const others = useMemo(() => detail.models.filter((m) => !m.baseline), [detail.models])
  const scenarios = useMemo(() => scenarioIds(detail), [detail])
  const byKey = useMemo(() => {
    const map = new Map<string, ClosedLoopEpisode>()
    detail.episodes.forEach((e) => map.set(`${e.model}\u0000${e.scenario}`, e))
    return map
  }, [detail.episodes])

  const option = useMemo<EChartsOption>(
    () => ({
      animation: false,
      tooltip: { ...modernTooltip, axisPointer: { type: 'shadow' } },
      legend: { ...baseLegend, type: 'scroll' },
      grid: plainGrid,
      xAxis: { type: 'category', data: scenarios, ...xAxisName('场景') },
      yAxis: { type: 'value', scale: false, splitLine: baseSplitLine, name: def.unit },
      series: others.map(
        (m): SeriesOption => ({
          name: m.id,
          type: 'bar',
          barMaxWidth: 22,
          itemStyle: { color: RANK_COLORS[detail.models.indexOf(m) % RANK_COLORS.length] },
          data: scenarios.map((s) => byKey.get(`${m.id}\u0000${s}`)?.difference?.[metric] ?? null),
        }),
      ),
    }),
    [others, scenarios, byKey, metric, def.unit, detail.models],
  )

  if (!others.length || !detail.baseline) return null
  return (
    <section className="page-section">
      <SectionHeader
        title="相对基线的逐场景配对差"
        description={`同一场景下，模型的 ${def.label} 减去基线 ${detail.baseline} 的值${
          detail.higher_is_better.includes(metric) ? '；正值为更好' : '；负值为更好'
        }`}
        extra={
          <Select
            aria-label="配对差的指标"
            value={metric}
            onChange={setMetric}
            style={{ minWidth: 140 }}
            options={METRICS.map((m) => ({ value: m.key, label: m.label }))}
          />
        }
      />
      <ChartPanel option={option} height={320} />
      <Panel style={{ marginTop: 12 }}>
        <Table<ClosedLoopModel>
          rowKey="id"
          size="middle"
          pagination={false}
          scroll={{ x: 'max-content' }}
          dataSource={others}
          columns={[
            { title: '模型', dataIndex: 'id' },
            ...METRICS.slice(0, 5).map((m) => ({
              title: `Δ ${m.label}`,
              align: 'right' as const,
              render: (_: unknown, row: ClosedLoopModel) => {
                const stat = row.paired?.[m.key]
                if (!stat || stat.mean === null) return '—'
                const kind = direction(m.key, stat.mean, detail.higher_is_better)
                return (
                  <span className="nowrap" style={{ color: DIRECTION_COLOR[kind] }} title={`${stat.n ?? '—'} 个配对场景`}>
                    {signed(stat.mean, m.digits)}
                    <span style={{ color: 'var(--text-secondary)' }}> ± {fmtNum(stat.std, m.digits)}</span>
                  </span>
                )
              },
            })),
          ]}
        />
      </Panel>
    </section>
  )
}

function traceOptions(trace: ClosedLoopTrace): { cte: EChartsOption; steering: EChartsOption; path: EChartsOption } {
  const s = trace.series
  const threshold = trace.cte_threshold
  const time = s.elapsed_s
  const points = (ys: (number | null)[]) => time.map((t, i): [number | null, number | null] => [t, ys[i]])
  const end = Math.ceil(Math.max(0, ...time.map((t) => t ?? 0)))
  const limit =
    threshold === null
      ? undefined
      : {
          silent: true,
          symbol: 'none',
          lineStyle: { color: OUTSIDE, type: 'dashed' as const, width: 1 },
          label: { formatter: '出界阈值', color: OUTSIDE },
          data: [{ yAxis: threshold }, { yAxis: -threshold }],
        }
  const cte: EChartsOption = {
    animation: false,
    tooltip: modernTooltip,
    grid: plainGrid,
    xAxis: { type: 'value', ...xAxisName('时间 (s)'), min: 0, max: end },
    yAxis: { type: 'value', scale: true, splitLine: baseSplitLine },
    series: [
      {
        name: 'CTE',
        type: 'line',
        showSymbol: false,
        data: points(s.cte),
        lineStyle: { width: 1.6, color: RANK_COLORS[0] },
        itemStyle: { color: RANK_COLORS[0] },
        markLine: limit,
      },
      {
        name: '出界',
        type: 'scatter',
        symbolSize: 5,
        itemStyle: { color: OUTSIDE },
        data: points(s.cte).filter((_, i) => s.outside[i]),
      },
    ],
  }
  const steering: EChartsOption = {
    animation: false,
    tooltip: modernTooltip,
    legend: baseLegend,
    grid: plainGrid,
    xAxis: { type: 'value', ...xAxisName('时间 (s)'), min: 0, max: end },
    // The legend names the axes, so the axes carry no name of their own (it would sit under the legend).
    yAxis: [
      { type: 'value', scale: true, splitLine: baseSplitLine, min: -1, max: 1 },
      { type: 'value', scale: true, splitLine: { show: false } },
    ],
    series: [
      { name: '转向（左轴）', type: 'line', showSymbol: false, data: points(s.steering), lineStyle: { width: 1.4, color: RANK_COLORS[1] }, itemStyle: { color: RANK_COLORS[1] } },
      { name: '速度（右轴）', type: 'line', showSymbol: false, yAxisIndex: 1, data: points(s.speed), lineStyle: { width: 1.2, color: RANK_COLORS[2] }, itemStyle: { color: RANK_COLORS[2] } },
    ],
  }
  // Top view of the drive (simulator x/z plane), coloured by |CTE|; red marks frames beyond the threshold.
  const track = s.x
    .map((x, i): [number | null, number | null, number] => [x, s.z[i], Math.abs(s.cte[i] ?? 0)])
    .filter((p) => p[0] !== null && p[1] !== null)
  const maxCte = Math.max(threshold ?? 0, ...track.map((p) => p[2]), 1e-6)
  // Same length per pixel on both axes, so the road keeps its shape: the plot area is PATH_ASPECT times as
  // wide as it is tall (the panel's width is capped in the stylesheet to match).
  const xs = track.map((p) => p[0] as number)
  const zs = track.map((p) => p[1] as number)
  const [x0, x1, z0, z1] = track.length ? [Math.min(...xs), Math.max(...xs), Math.min(...zs), Math.max(...zs)] : [0, 1, 0, 1]
  const span = Math.max((x1 - x0) / PATH_ASPECT, z1 - z0, 1) * 1.1
  const range = (low: number, high: number, width: number) => {
    const centre = (low + high) / 2
    return { min: Math.floor(centre - width / 2), max: Math.ceil(centre + width / 2) }
  }
  const path: EChartsOption = {
    animation: false,
    tooltip: {
      trigger: 'item',
      formatter: (item) => {
        const value = (item as { value: number[] }).value
        return `x ${value[0].toFixed(1)} · z ${value[1].toFixed(1)}<br/>|CTE| ${value[2].toFixed(3)}`
      },
    },
    grid: { left: 56, right: 88, top: 24, bottom: 44 },
    xAxis: { type: 'value', ...range(x0, x1, span * PATH_ASPECT), ...xAxisName('x'), splitLine: baseSplitLine },
    yAxis: { type: 'value', ...range(z0, z1, span), name: 'z', splitLine: baseSplitLine },
    visualMap: {
      type: 'continuous',
      dimension: 2,
      min: 0,
      max: maxCte,
      seriesIndex: 0,
      right: 8,
      top: 'middle',
      text: ['|CTE|', ''],
      calculable: false,
      inRange: { color: ['#2f9e7a', '#d4b02a', '#d64545'] },
      formatter: (value) => Number(value).toFixed(2),
    },
    series: [
      { name: '轨迹', type: 'scatter', symbolSize: 4, data: track, large: false },
      {
        name: '起点',
        type: 'scatter',
        symbol: 'pin',
        symbolSize: 26,
        itemStyle: { color: '#18181b' },
        data: track.length ? [[track[0][0], track[0][1], track[0][2]]] : [],
        tooltip: { formatter: '起点' },
      },
      {
        name: '出界或碰撞',
        type: 'scatter',
        symbol: 'diamond',
        symbolSize: 9,
        itemStyle: { color: OUTSIDE, borderColor: '#fff', borderWidth: 1 },
        data: s.x
          .map((x, i): [number | null, number | null, number, boolean] => [x, s.z[i], Math.abs(s.cte[i] ?? 0), s.outside[i] || s.hit[i]])
          .filter((p) => p[3] && p[0] !== null && p[1] !== null)
          .map((p) => [p[0], p[1], p[2]]),
      },
    ],
  }
  return { cte, steering, path }
}

function TraceSection({ detail }: { detail: ClosedLoopDetail }) {
  const models = detail.models.map((m) => m.id)
  const scenarios = scenarioIds(detail)
  const [model, setModel] = useState(models[0])
  const [scenario, setScenario] = useState(scenarios[0])
  const activeModel = models.includes(model) ? model : models[0]
  const activeScenario = scenarios.includes(scenario) ? scenario : scenarios[0]
  const trace = useQuery({
    queryKey: ['closed-loop-trace', detail.id, activeModel, activeScenario],
    queryFn: () => api.closedLoopTrace(detail.id, activeModel, activeScenario),
    enabled: detail.has_frames && !!activeModel && !!activeScenario,
    staleTime: Infinity,
  })
  const episode = detail.episodes.find((e) => e.model === activeModel && e.scenario === activeScenario)
  const targetSpeed = num(episode?.target_speed_mph) ?? detail.target_speed_mph
  const options = useMemo(() => (trace.data ? traceOptions(trace.data) : null), [trace.data])

  return (
    <section className="page-section">
      <SectionHeader
        title="单次运行"
        description={
          episode
            ? `${TERMINATION[String(episode.termination)] ?? episode.termination ?? '—'}结束 · 存活 ${fmtNum(num(episode.survival_s), 1)} s · 平均 |CTE| ${fmtNum(num(episode.mean_abs_cte), 3)} · 出界 ${episode.excursions ?? '—'} 次`
            : '选择模型与场景查看逐帧记录'
        }
        extra={
          <span style={{ display: 'inline-flex', gap: 8, flexWrap: 'wrap' }}>
            <Select aria-label="模型" value={activeModel} onChange={setModel} style={{ minWidth: 180 }}
              options={models.map((m) => ({ value: m, label: m }))} />
            <Select aria-label="场景" value={activeScenario} onChange={setScenario} style={{ minWidth: 120 }}
              options={scenarios.map((s) => ({ value: s, label: s }))} />
          </span>
        }
      />
      {!detail.has_frames ? (
        <Panel><EmptyState compact title="这次评估没有保存逐帧记录（frames.jsonl）" /></Panel>
      ) : trace.isLoading ? (
        <Panel padded><Spin /></Panel>
      ) : trace.error ? (
        <Alert type="warning" showIcon message={String((trace.error as Error).message ?? trace.error)} />
      ) : options && trace.data ? (
        <>
          <div className="closed-loop-charts">
            <ChartPanel
              title={cameraCte(detail) ? '横向位置（图像估计）' : '横向偏差 CTE'}
              subtitle={
                trace.data.outside_frames
                  ? `${trace.data.frames} 帧中有 ${trace.data.outside_frames} 帧超过阈值`
                  : `${trace.data.cte_frames ?? trace.data.frames} / ${trace.data.frames} 帧有 CTE 测量，均在阈值内`
              }
              option={options.cte}
              height={280}
            />
            <ChartPanel
              title="转向与速度"
              subtitle={
                targetSpeed
                  ? `模型输出的转向；油门由速度控制器给出（目标 ${targetSpeed} mph）`
                  : '模型输出的转向（固定油门）'
              }
              option={options.steering}
              height={280}
            />
          </div>
          {trace.data.has_positions === false ? (
            <Panel>
              <EmptyState compact title="没有俯视轨迹" description="这个模拟器不回传车辆位置；行驶距离由速度积分得到。" />
            </Panel>
          ) : (
            <ChartPanel
              title="俯视轨迹"
              subtitle={`模拟器 x–z 平面，按 |CTE| 着色${trace.data.collision_frames ? `；${trace.data.collision_frames} 帧发生碰撞` : ''}`}
              option={options.path}
              height={420}
              className="closed-loop-path"
            />
          )}
        </>
      ) : null}
    </section>
  )
}

export default function ClosedLoop() {
  const [params, setParams] = useSearchParams()
  const list = useQuery({ queryKey: ['closed-loops'], queryFn: api.closedLoops })
  const usable = (list.data ?? []).filter((e) => e.status !== 'error')
  const selected = usable.find((e) => e.id === params.get('eval'))?.id ?? usable[0]?.id
  const detail = useQuery({
    queryKey: ['closed-loop', selected],
    queryFn: () => api.closedLoop(selected as string),
    enabled: !!selected,
  })

  if (list.isLoading) return <Spin />
  if (list.error) return <Alert type="error" showIcon message={String((list.error as Error).message ?? list.error)} />
  if (!usable.length) {
    return (
      <Panel>
        <EmptyState
          title="没有闭环评估归档"
          description="运行 python -m apps.eval.closed_loop 后，结果目录放在 docs/experiments/semester_2026_fall/closed_loop/ 下即可在此查看。"
        />
      </Panel>
    )
  }
  const data = detail.data
  const rules = data?.rules ?? {}
  return (
    <>
      <section className="page-section" style={{ marginTop: 0 }}>
        <SectionHeader
          title="闭环评估"
          description={
            data
              ? `${data.environment ?? '模拟器'} · ${data.scenarios} 个场景 × ${data.duration_s ?? '—'} s · ${
                  data.target_speed_mph ? `默认目标车速 ${data.target_speed_mph} mph` : `固定油门 ${data.fixed_throttle ?? '—'}`
                } · 出界阈值 |CTE| > ${rules.cte_threshold ?? '—'}（连续 ${rules.consecutive_frames ?? '—'} 帧）${
                  cameraCte(data) ? ' · CTE 为前视图像估计的横向位置，不是模拟器真值' : ''
                }`
              : '读取归档的模拟器闭环评估结果'
          }
          extra={
            <Segmented
              aria-label="闭环评估归档"
              value={selected}
              onChange={(v) =>
                setParams((current) => {
                  const next = new URLSearchParams(current)
                  next.set('eval', String(v))
                  return next
                })
              }
              options={usable.map((e) => ({ value: e.id, label: e.id }))}
            />
          }
        />
        {detail.isLoading || !data ? (
          <Panel padded>{detail.error ? <Alert type="error" showIcon message={String((detail.error as Error).message)} /> : <Spin />}</Panel>
        ) : (
          <>
            <Panel>
              <ModelTable detail={data} />
            </Panel>
            <div className="section-header-desc" style={{ marginTop: 8 }}>
              完成于 {fmtTime(data.finished_unix_s)} · 设备 {data.device ?? '—'} · 起点：{data.start_control ?? '—'}。
              不同模拟器会话的道路可能不同，跨归档的数值不能直接比较；同一归档内各模型在相同场景上配对。
            </div>
          </>
        )}
      </section>
      {data ? <PairedSection key={`paired-${data.id}`} detail={data} /> : null}
      {data ? <TraceSection key={`trace-${data.id}`} detail={data} /> : null}
    </>
  )
}
