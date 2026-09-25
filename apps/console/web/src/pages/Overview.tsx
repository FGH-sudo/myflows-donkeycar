import { useQuery, useQueryClient } from '@tanstack/react-query'
import { App, Alert, Button, Card, Col, Empty, Progress, Row, Space, Table, Tag, Tooltip } from 'antd'
import type { EChartsOption } from 'echarts'
import { useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api'
import type { GpuInfo, GpuPayload, GpuSample, Job } from '../api'
import EChart from '../components/EChart'
import MetricCard from '../components/MetricCard'
import StatusTag from '../components/StatusTag'
import { fmtBytes, fmtDuration, fmtTime, isActive } from '../format'
import { useEventSource } from '../hooks/useEventSource'

const HISTORY_S = 300

function pctColor(value: number | null | undefined) {
  if (value === null || value === undefined) return '#bfbfbf'
  if (value > 85) return '#f5222d'
  if (value > 60) return '#fa8c16'
  return '#1677ff'
}

function GpuCard({ gpu, history }: { gpu: GpuInfo; history: GpuSample[] }) {
  const memPct = gpu.memory_used_mib !== null && gpu.memory_total_mib ? (gpu.memory_used_mib / gpu.memory_total_mib) * 100 : null
  const option = useMemo<EChartsOption>(() => {
    const t0 = history.length ? history[history.length - 1].unix_s : 0
    const pick = (fn: (g: GpuInfo) => number | null) =>
      history.map((s) => {
        const g = s.gpus.find((x) => x.index === gpu.index)
        return [s.unix_s - t0, g ? fn(g) : null]
      })
    return {
      animation: false,
      grid: { left: 40, right: 44, top: 28, bottom: 24 },
      legend: { top: 0, right: 0, textStyle: { fontSize: 11 }, itemWidth: 14, itemHeight: 8 },
      tooltip: { trigger: 'axis', valueFormatter: (v) => (typeof v === 'number' ? v.toFixed(1) : String(v)) },
      xAxis: { type: 'value', min: -HISTORY_S, max: 0, axisLabel: { formatter: (v: number) => `${v}s`, fontSize: 10 } },
      yAxis: [
        { type: 'value', min: 0, max: 100, axisLabel: { formatter: '{value}%', fontSize: 10 } },
        { type: 'value', min: 0, axisLabel: { formatter: '{value}W', fontSize: 10 }, splitLine: { show: false } },
      ],
      series: [
        { name: '利用率', type: 'line', showSymbol: false, areaStyle: { opacity: 0.15 }, data: pick((g) => g.utilization_pct) },
        { name: '显存', type: 'line', showSymbol: false, data: pick((g) => (g.memory_used_mib !== null && g.memory_total_mib ? (g.memory_used_mib / g.memory_total_mib) * 100 : null)) },
        { name: '功耗', type: 'line', showSymbol: false, yAxisIndex: 1, lineStyle: { type: 'dashed' }, data: pick((g) => g.power_w) },
      ],
    }
  }, [history, gpu.index])

  const missing = Object.entries(gpu.missing ?? {})
  return (
    <Card title={<span>GPU {gpu.index} · {gpu.name}</span>} size="small"
      extra={missing.length ? (
        <Tooltip title={missing.map(([k, v]) => `${k}: ${v}`).join('\n')}><Tag color="warning">{missing.length} 项缺测</Tag></Tooltip>
      ) : <Tag color="success">采集完整</Tag>}>
      <Row gutter={16} align="middle">
        <Col xs={24} md={9}>
          <Space size={18} wrap>
            <Progress type="dashboard" percent={Math.round(gpu.utilization_pct ?? 0)} size={108}
              strokeColor={pctColor(gpu.utilization_pct)} format={(p) => (gpu.utilization_pct === null ? '—' : `${p}%`)} />
            <div style={{ minWidth: 150 }}>
              <div className="metric-label">显存</div>
              <div style={{ fontWeight: 600 }}>
                {gpu.memory_used_mib !== null ? `${(gpu.memory_used_mib / 1024).toFixed(2)} / ${((gpu.memory_total_mib ?? 0) / 1024).toFixed(1)} GB` : '—'}
              </div>
              <Progress percent={Math.round(memPct ?? 0)} size="small" showInfo={false} strokeColor={pctColor(memPct)} />
              <div className="metric-label" style={{ marginTop: 6 }}>温度 / 功耗</div>
              <div style={{ fontWeight: 600 }}>
                {gpu.temperature_c !== null ? `${gpu.temperature_c.toFixed(0)} °C` : '—'}
                <span className="muted"> · </span>
                {gpu.power_w !== null ? `${gpu.power_w.toFixed(1)} W` : '—'}
                {gpu.power_limit_w ? <span className="muted"> / {gpu.power_limit_w.toFixed(0)} W</span> : null}
              </div>
            </div>
          </Space>
        </Col>
        <Col xs={24} md={15}>
          <EChart option={option} height={170} />
        </Col>
      </Row>
    </Card>
  )
}

function GpuProcesses({ gpu }: { gpu: GpuInfo }) {
  const rows = (gpu.processes ?? []).filter((p) => p.owner)
  const others = (gpu.processes ?? []).length - rows.length
  return (
    <Card size="small" title="训练进程占用" extra={<span className="muted">其他 GPU 进程 {others} 个</span>}>
      {rows.length ? (
        <Table size="small" rowKey="pid" pagination={false} dataSource={rows}
          columns={[
            { title: 'PID', dataIndex: 'pid', width: 90 },
            { title: '任务', render: (_, p) => <Link to={`/run/console/${p.owner!.job_id}`}>{p.owner!.label}</Link> },
            { title: 'rank', render: (_, p) => (p.owner?.rank !== undefined ? `rank ${p.owner.rank}` : '启动器/PS') , width: 110 },
            { title: '显存', render: (_, p) => (p.used_memory_mib !== null ? `${p.used_memory_mib.toFixed(0)} MiB` : <Tooltip title={gpu.missing?.process_memory}>不可读</Tooltip>), width: 100 },
          ]} />
      ) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="当前没有控制台任务使用 GPU" />}
    </Card>
  )
}

function ActiveJobs({ jobs }: { jobs: Job[] }) {
  const { message } = App.useApp()
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const active = jobs.filter((j) => isActive(j.status))
  const recent = jobs.filter((j) => !isActive(j.status)).slice(0, 5)
  const stop = async (id: string) => {
    try {
      await api.stopJob(id)
      message.success('已发送停止请求')
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
    } catch (e) {
      message.error(String((e as Error).message))
    }
  }
  const columns = [
    { title: '任务', dataIndex: 'label', render: (v: string, j: Job) => <Link to={`/run/console/${j.id}`}>{v}</Link> },
    { title: '状态', dataIndex: 'status', width: 90, render: (s: string) => <StatusTag status={s} /> },
    { title: '开始', dataIndex: 'started_unix_s', width: 130, render: (v: number | null, j: Job) => fmtTime(v ?? j.created_unix_s) },
    { title: '耗时', width: 100, render: (_: unknown, j: Job) => fmtDuration(j.started_unix_s ? (j.finished_unix_s ?? Date.now() / 1000) - j.started_unix_s : null) },
    { title: '', width: 80, render: (_: unknown, j: Job) => (isActive(j.status) ? <Button size="small" danger onClick={() => stop(j.id)}>停止</Button> : null) },
  ]
  return (
    <Card size="small" title="控制台任务" extra={<Button type="primary" size="small" onClick={() => navigate('/new')}>新建训练</Button>}>
      <Table size="small" rowKey="id" pagination={false} columns={columns} dataSource={[...active, ...recent]}
        locale={{ emptyText: '还没有从控制台发起的任务' }} />
    </Card>
  )
}

export default function Overview() {
  const initial = useQuery({ queryKey: ['gpu'], queryFn: api.gpu })
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: api.jobs, refetchInterval: 3000 })
  const [live, setLive] = useState<{ latest: GpuSample | null; history: GpuSample[] } | null>(null)

  const base: GpuPayload | undefined = initial.data
  const history = live?.history ?? base?.history ?? []
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

  return (
    <div>
      <div className="page-title">
        <h2>总览</h2>
        <span className="muted">{connected ? <><span className="live-dot" />实时 · 每 {base?.interval_s ?? 1} 秒采样 · {base?.backend}</> : '连接中…'}</span>
      </div>
      {base && !base.backend ? (
        <Alert type="error" showIcon style={{ marginBottom: 16 }} message="GPU 采样不可用"
          description={Object.entries(base.backend_errors).map(([k, v]) => `${k}: ${v}`).join('；')} />
      ) : null}
      <Row gutter={[16, 16]}>
        <Col xs={12} lg={6}><MetricCard label="CPU 利用率" value={latest ? latest.cpu_utilization_pct.toFixed(0) : '—'} unit="%" /></Col>
        <Col xs={12} lg={6}><MetricCard label="内存" value={latest ? fmtBytes(latest.ram_used_bytes) : '—'} unit={latest ? `/ ${fmtBytes(latest.ram_total_bytes)}` : ''} /></Col>
        <Col xs={12} lg={6}><MetricCard label="GPU 数量" value={latest?.gpus.length ?? '—'} unit={base?.driver_version ? `驱动 ${base.driver_version}` : ''} /></Col>
        <Col xs={12} lg={6}><MetricCard label="进行中任务" value={(jobs.data ?? []).filter((j) => isActive(j.status)).length} unit="个（单 GPU 串行）" /></Col>
        {(latest?.gpus ?? []).map((gpu) => (
          <Col span={24} key={gpu.index}><GpuCard gpu={gpu} history={history} /></Col>
        ))}
        <Col xs={24} xl={14}><ActiveJobs jobs={jobs.data ?? []} /></Col>
        <Col xs={24} xl={10}>{latest?.gpus[0] ? <GpuProcesses gpu={latest.gpus[0]} /> : null}</Col>
      </Row>
    </div>
  )
}
