import { App, Button, Table, Tooltip } from 'antd'
import { PlusOutlined, RightOutlined, RocketOutlined, StopOutlined } from '@ant-design/icons'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import type { EChartsOption } from 'echarts'
import * as echarts from 'echarts'
import { useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api'
import type { GpuInfo, GpuPayload, GpuSample, Job } from '../api'
import ChartPanel from '../components/ChartPanel'
import EmptyState from '../components/EmptyState'
import { Panel, SectionHeader } from '../components/Panel'
import StatStrip from '../components/StatStrip'
import StatusTag from '../components/StatusTag'
import { fmtDuration, fmtTime, isActive, rankColor, shortGpuName } from '../format'
import { useEventSource } from '../hooks/useEventSource'

const HISTORY_S = 300
const GIB = 1024 ** 3

const COLORS = { gpu: '#3b6fd8', vram: '#8866d6', cpu: '#2f9e7a', ram: '#d4892a', power: '#7a8394' }

const memPct = (g: GpuInfo | undefined) =>
  g && g.memory_used_mib !== null && g.memory_total_mib ? (g.memory_used_mib / g.memory_total_mib) * 100 : null

function summarize(values: (number | null)[]) {
  const v = values.filter((x): x is number => x !== null && Number.isFinite(x))
  if (!v.length) return null
  return { mean: v.reduce((a, b) => a + b, 0) / v.length, max: Math.max(...v) }
}

const trendHint = (values: (number | null)[]) => {
  const s = summarize(values)
  return s ? `5 分钟均值 ${s.mean.toFixed(0)}% · 峰值 ${s.max.toFixed(0)}%` : '等待采样…'
}

function GpuSection({ gpu, history, payload }: { gpu: GpuInfo; history: GpuSample[]; payload: GpuPayload | undefined }) {
  const option = useMemo<EChartsOption>(() => {
    const t0 = history.length ? history[history.length - 1].unix_s : 0
    const pick = (fn: (g: GpuInfo) => number | null) =>
      history.map((s) => {
        const g = s.gpus.find((x) => x.index === gpu.index)
        return [s.unix_s - t0, g ? fn(g) : null]
      })
    return {
      animation: false,
      grid: { left: 44, right: 52, top: 40, bottom: 28 },
      legend: { top: 4, left: 4 },
      tooltip: { trigger: 'axis', valueFormatter: (v) => (typeof v === 'number' ? v.toFixed(1) : String(v)) },
      xAxis: {
        type: 'value',
        min: -HISTORY_S,
        max: 0,
        interval: 60,
        axisLabel: { formatter: (v: number) => (v === 0 ? '现在' : `${v}s`) },
      },
      yAxis: [
        { type: 'value', min: 0, max: 100, interval: 25, axisLabel: { formatter: '{value}%' } },
        { type: 'value', min: 0, axisLabel: { formatter: '{value} W' }, splitLine: { show: false } },
      ],
      series: [
        {
          name: 'GPU 负载',
          type: 'line',
          showSymbol: false,
          lineStyle: { width: 1.6, color: COLORS.gpu },
          itemStyle: { color: COLORS.gpu },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(59, 111, 216, 0.14)' },
              { offset: 1, color: 'rgba(59, 111, 216, 0)' },
            ]),
          },
          data: pick((g) => g.utilization_pct),
        },
        {
          name: '显存占用',
          type: 'line',
          showSymbol: false,
          lineStyle: { width: 1.6, color: COLORS.vram },
          itemStyle: { color: COLORS.vram },
          data: pick((g) => memPct(g)),
        },
        {
          name: '功耗',
          type: 'line',
          showSymbol: false,
          yAxisIndex: 1,
          lineStyle: { type: 'dashed', width: 1.2, color: COLORS.power },
          itemStyle: { color: COLORS.power },
          data: pick((g) => g.power_w),
        },
      ],
    }
  }, [history, gpu.index])

  const missing = Object.entries(gpu.missing ?? {})

  return (
    <section className="page-section">
      <SectionHeader
        title="GPU 遥测"
        description={[`GPU ${gpu.index}`, shortGpuName(gpu.name), payload?.driver_version ? `驱动 ${payload.driver_version}` : null]
          .filter(Boolean)
          .join(' · ')}
        extra={
          <div className="inline-stats">
            <span>
              温度<b>{gpu.temperature_c !== null ? `${gpu.temperature_c.toFixed(0)} °C` : '—'}</b>
            </span>
            <span>
              功耗<b>{gpu.power_w !== null ? `${gpu.power_w.toFixed(1)} W` : '—'}</b>
              {gpu.power_limit_w ? <span className="muted"> / {gpu.power_limit_w.toFixed(0)} W</span> : null}
            </span>
            {payload?.backend ? (
              <span className="hide-sm">
                采样<b>{payload.backend.toUpperCase()}</b>
              </span>
            ) : null}
            {missing.length ? (
              <Tooltip title={missing.map(([k, v]) => `${k}: ${v}`).join('\n')}>
                <span className="toolbar-item" style={{ gap: 6, fontSize: 13, cursor: 'help' }}>
                  <span className="status-dot warn" />
                  {missing.length} 项缺测
                </span>
              </Tooltip>
            ) : null}
          </div>
        }
      />
      <ChartPanel option={option} height={260} />
    </section>
  )
}

function QueueSection({ jobs, gpu }: { jobs: Job[]; gpu: GpuInfo | undefined }) {
  const { message } = App.useApp()
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const active = jobs.filter((j) => isActive(j.status))
  const pending = jobs.filter((j) => j.status === 'pending').length
  const recent = jobs.filter((j) => !isActive(j.status)).slice(0, 6)
  const processes = (gpu?.processes ?? []).filter((p) => p.owner)
  const others = (gpu?.processes ?? []).length - processes.length

  const stop = async (id: string) => {
    try {
      await api.stopJob(id)
      message.success('已发送停止请求')
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
    } catch (e) {
      message.error(String((e as Error).message))
    }
  }

  return (
    <section className="page-section">
      <SectionHeader
        title="训练队列"
        description={`${active.length} 个进行中 · ${pending} 个排队 · 单卡按提交顺序串行执行`}
        extra={
          <Link to="/runs?source=console" className="row-action">
            全部记录
            <RightOutlined />
          </Link>
        }
      />
      <Panel>
        {jobs.length ? (
          <Table<Job>
            size="middle"
            rowKey="id"
            pagination={false}
            dataSource={[...active, ...recent]}
            columns={[
              {
                title: '任务',
                dataIndex: 'label',
                ellipsis: { showTitle: false },
                render: (v: string, j) => (
                  <Tooltip title={v} placement="topLeft">
                    <Link to={`/run/console/${j.id}`} className="table-link">
                      {v}
                    </Link>
                  </Tooltip>
                ),
              },
              { title: '状态', dataIndex: 'status', width: 110, render: (s: string) => <StatusTag status={s} /> },
              {
                title: '开始时间',
                dataIndex: 'started_unix_s',
                width: 150,
                responsive: ['sm'],
                render: (v: number | null, j) => <span className="num muted">{fmtTime(v ?? j.created_unix_s)}</span>,
              },
              {
                title: '耗时',
                width: 110,
                align: 'right',
                render: (_, j) => (
                  <span className="num">
                    {fmtDuration(j.started_unix_s ? (j.finished_unix_s ?? Date.now() / 1000) - j.started_unix_s : null)}
                  </span>
                ),
              },
              ...(active.length
                ? [
                    {
                      title: '',
                      width: 84,
                      align: 'right' as const,
                      render: (_: unknown, j: Job) =>
                        isActive(j.status) ? (
                          <Button size="small" danger type="text" icon={<StopOutlined />} onClick={() => stop(j.id)}>
                            停止
                          </Button>
                        ) : null,
                    },
                  ]
                : []),
            ]}
          />
        ) : (
          <EmptyState
            icon={<RocketOutlined />}
            title="还没有训练任务"
            description="从控制台发起的分布式或单进程训练会出现在这里"
            action={
              <Button type="primary" icon={<PlusOutlined />} onClick={() => navigate('/new')}>
                新建训练
              </Button>
            }
          />
        )}

        {processes.length ? (
          <>
            <hr className="panel-divider" />
            <div className="panel-head" style={{ paddingBottom: 4 }}>
              <div className="panel-title">GPU 训练进程</div>
              <span className="muted" style={{ fontSize: 12 }}>另有 {others} 个系统进程</span>
            </div>
            <Table
              size="small"
              rowKey="pid"
              pagination={false}
              dataSource={processes}
              columns={[
                { title: 'PID', dataIndex: 'pid', width: 100, render: (v) => <span className="mono">{v}</span> },
                {
                  title: '任务',
                  ellipsis: true,
                  render: (_, p) => (
                    <Link to={`/run/console/${p.owner!.job_id}`} className="table-link">
                      {p.owner!.label}
                    </Link>
                  ),
                },
                {
                  title: '角色',
                  width: 120,
                  render: (_, p) =>
                    p.owner?.rank !== undefined ? (
                      <span className="nowrap" style={{ fontWeight: 600, color: rankColor(p.owner.rank) }}>
                        rank {p.owner.rank}
                      </span>
                    ) : (
                      <span className="muted nowrap">启动器 / PS</span>
                    ),
                },
                {
                  title: '显存',
                  width: 120,
                  align: 'right',
                  render: (_, p) =>
                    p.used_memory_mib !== null ? (
                      <span className="num">{p.used_memory_mib.toFixed(0)} MiB</span>
                    ) : (
                      <Tooltip title={gpu?.missing?.process_memory}>
                        <span className="muted nowrap">WDDM 不可读</span>
                      </Tooltip>
                    ),
                },
              ]}
            />
          </>
        ) : (
          <div className="panel-footer">
            <span className="toolbar-item" style={{ gap: 8, color: 'var(--text-muted)' }}>
              <span className="status-dot" />
              当前没有训练进程占用 GPU
            </span>
            {gpu ? <span className="hide-sm">另有 {others} 个系统进程</span> : null}
          </div>
        )}
      </Panel>
    </section>
  )
}

export default function Overview() {
  const initial = useQuery({ queryKey: ['gpu'], queryFn: api.gpu })
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: api.jobs, refetchInterval: 3000 })
  const [live, setLive] = useState<{ latest: GpuSample | null; history: GpuSample[] } | null>(null)

  const base: GpuPayload | undefined = initial.data
  const history = useMemo(() => live?.history ?? base?.history ?? [], [live, base])
  const latest = live?.latest ?? base?.latest ?? null

  const { connected } = useEventSource(base ? '/api/gpu/stream' : null, {
    gpu: (data) => {
      const payload = data as GpuPayload
      if (!payload.latest) return
      setLive((prev) => {
        const hist = [...(prev?.history ?? base?.history ?? []), payload.latest!]
        const cutoff = payload.latest!.unix_s - HISTORY_S
        return { latest: payload.latest, history: hist.filter((s) => s.unix_s >= cutoff) }
      })
    },
  })

  const gpus = latest?.gpus ?? []
  const gpu0 = gpus[0]
  const trend = useMemo(
    () => ({
      gpu: history.map((s) => s.gpus[0]?.utilization_pct ?? null),
      vram: history.map((s) => memPct(s.gpus[0])),
      cpu: history.map((s) => s.cpu_utilization_pct),
      ram: history.map((s) => (s.ram_total_bytes > 0 ? (s.ram_used_bytes / s.ram_total_bytes) * 100 : null)),
    }),
    [history],
  )

  return (
    <div>
      <div className="page-title">
        <div>
          <h2>系统总览</h2>
          <div className="page-subtitle">实时硬件遥测与训练队列</div>
        </div>
        <span className="status-chip">
          {connected ? (
            <>
              <span className="live-dot" style={{ width: 6, height: 6 }} />
              <span style={{ color: 'var(--text-main)' }}>实时</span>
              <span className="muted">每 {base?.interval_s ?? 1} 秒采样</span>
            </>
          ) : (
            <>
              <span className="status-dot warn" />
              连接采样流…
            </>
          )}
        </span>
      </div>

      {base && !base.backend ? (
        <Panel padded style={{ marginBottom: 20, borderColor: 'rgba(209, 67, 67, 0.35)' }}>
          <span className="toolbar-item" style={{ color: 'var(--red)', fontWeight: 500 }}>
            <span className="status-dot err" />
            GPU 采样不可用
          </span>
          <div className="muted" style={{ fontSize: 13, marginTop: 4 }}>
            {Object.entries(base.backend_errors).map(([k, v]) => `${k}: ${v}`).join('；')}
          </div>
        </Panel>
      ) : null}

      <StatStrip
        items={[
          {
            label: 'GPU 负载',
            color: COLORS.gpu,
            value: gpu0?.utilization_pct !== null && gpu0?.utilization_pct !== undefined ? gpu0.utilization_pct.toFixed(0) : '—',
            unit: '%',
            spark: trend.gpu,
            sparkRange: [0, 100],
            hint: trendHint(trend.gpu),
          },
          {
            label: '显存',
            color: COLORS.vram,
            value: gpu0?.memory_used_mib !== null && gpu0?.memory_used_mib !== undefined ? (gpu0.memory_used_mib / 1024).toFixed(2) : '—',
            unit: gpu0?.memory_total_mib ? `/ ${(gpu0.memory_total_mib / 1024).toFixed(1)} GB` : 'GB',
            spark: trend.vram,
            sparkRange: [0, 100],
            hint: trendHint(trend.vram),
          },
          {
            label: 'CPU',
            color: COLORS.cpu,
            value: latest ? latest.cpu_utilization_pct.toFixed(0) : '—',
            unit: '%',
            spark: trend.cpu,
            sparkRange: [0, 100],
            hint: trendHint(trend.cpu),
          },
          {
            label: '系统内存',
            color: COLORS.ram,
            value: latest ? (latest.ram_used_bytes / GIB).toFixed(1) : '—',
            unit: latest ? `/ ${(latest.ram_total_bytes / GIB).toFixed(1)} GB` : 'GB',
            spark: trend.ram,
            sparkRange: [0, 100],
            hint: trendHint(trend.ram),
          },
        ]}
      />

      {gpus.map((gpu) => (
        <GpuSection key={gpu.index} gpu={gpu} history={history} payload={base} />
      ))}

      <QueueSection jobs={jobs.data ?? []} gpu={gpu0} />
    </div>
  )
}
