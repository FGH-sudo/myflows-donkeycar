import {
  ArrowLeftOutlined,
  BranchesOutlined,
  CodeOutlined,
  FieldTimeOutlined,
  FileTextOutlined,
  FundViewOutlined,
  LineChartOutlined,
  StopOutlined,
} from '@ant-design/icons'
import { useQueryClient } from '@tanstack/react-query'
import { Alert, App, Button, Popconfirm, Spin, Tabs, Tooltip } from 'antd'
import type { ReactNode } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { api } from '../api'
import StatStrip from '../components/StatStrip'
import StatusTag from '../components/StatusTag'
import { KIND_TEXT, fmtDuration, fmtNum, fmtTime, headlineMetric, modeLabel } from '../format'
import CurvesTab from './run/CurvesTab'
import LogTab from './run/LogTab'
import ResourcesTab from './run/ResourcesTab'
import SummaryTab from './run/SummaryTab'
import TimingTab from './run/TimingTab'
import TopologyTab from './run/TopologyTab'
import { useRunData } from './run/useRunData'

const tabLabel = (icon: ReactNode, text: string) => (
  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
    {icon}
    {text}
  </span>
)

export default function RunDetailPage() {
  const params = useParams()
  const runId = params['*'] ?? ''
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const activeTab = searchParams.get('tab') || 'curves'
  const { message } = App.useApp()
  const queryClient = useQueryClient()
  const { detail, state, summary, rankStatus, ranks, live, connected } = useRunData(runId)

  if (detail.isLoading) return <Spin style={{ display: 'block', marginTop: 80 }} />
  if (detail.error || !detail.data) {
    return <Alert type="error" showIcon message="无法加载运行" description={String((detail.error as Error)?.message ?? '')} />
  }
  const data = { ...detail.data, summary: summary ?? detail.data.summary }
  const s = data.summary
  const jobId = s.job_id

  const stop = async () => {
    if (!jobId) return
    try {
      await api.stopJob(jobId)
      message.success('已发送停止请求')
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
    } catch (e) {
      message.error((e as Error).message)
    }
  }

  const elapsed = live && s.started_unix_s ? Date.now() / 1000 - s.started_unix_s : s.elapsed_s
  const items = [
    {
      key: 'curves',
      label: tabLabel(<LineChartOutlined />, '训练曲线'),
      children: <CurvesTab detail={data} state={state} ranks={ranks} />,
    },
    {
      key: 'timing',
      label: tabLabel(<FieldTimeOutlined />, '耗时拆分'),
      children: <TimingTab detail={data} state={state} ranks={ranks} />,
    },
    {
      key: 'resources',
      label: tabLabel(<FundViewOutlined />, '硬件资源'),
      children: <ResourcesTab detail={data} state={state} />,
    },
    ...(data.kind === 'distributed'
      ? [
          {
            key: 'topology',
            label: tabLabel(<BranchesOutlined />, '分布式拓扑'),
            children: <TopologyTab detail={data} rankStatus={rankStatus} live={live} />,
          },
        ]
      : []),
    {
      key: 'summary',
      label: tabLabel(<FileTextOutlined />, '配置与结果'),
      children: <SummaryTab detail={data} />,
    },
    {
      key: 'log',
      label: tabLabel(<CodeOutlined />, '终端日志'),
      children: <LogTab runId={runId} live={live} />,
    },
  ]

  const headline = headlineMetric(s)
  const meta = [
    KIND_TEXT[s.kind] ?? s.kind,
    modeLabel(s),
    data.kind === 'distributed' ? `${s.workers} Worker` : null,
    s.device,
    s.started_unix_s ? `开始于 ${fmtTime(s.started_unix_s)}` : null,
  ].filter(Boolean)

  return (
    <div>
      <div className="run-header">
        <div className="run-header-main">
          <Tooltip title="返回">
            <Button type="text" icon={<ArrowLeftOutlined />} onClick={() => navigate(-1)} aria-label="返回" className="run-back" />
          </Tooltip>
          <div style={{ minWidth: 0 }}>
            <div className="run-title-row">
              <h2 className="run-title">{s.label}</h2>
              <StatusTag status={s.status} />
              {live ? (
                <span className="toolbar-item" style={{ gap: 6 }}>
                  {connected ? (
                    <>
                      <span className="live-dot" />
                      <span style={{ color: 'var(--green)' }}>实时更新中</span>
                    </>
                  ) : (
                    <span className="muted">连接中…</span>
                  )}
                </span>
              ) : null}
            </div>
            <div className="run-meta">
              {meta.map((m, i) => (
                <span key={i}>{m}</span>
              ))}
            </div>
          </div>
        </div>

        {live && jobId ? (
          <Popconfirm
            title="停止该任务？"
            description="分布式任务会结束全部进程；单进程任务会先保存断点再退出。"
            onConfirm={stop}
            okText="停止"
            okButtonProps={{ danger: true }}
          >
            <Button danger icon={<StopOutlined />} disabled={s.status === 'stopping'}>
              {s.status === 'stopping' ? '正在停止…' : '停止任务'}
            </Button>
          </Popconfirm>
        ) : null}
      </div>

      <div style={{ marginBottom: 'calc(28px * var(--s))' }}>
        <StatStrip
          items={[
            { label: headline?.label ?? '核心指标', value: headline?.value ?? '—', hint: headline ? '最终验证结果' : '运行结束后显示' },
            {
              label: '吞吐',
              value: s.final?.samples_per_s ? fmtNum(s.final.samples_per_s, 0) : '—',
              unit: s.final?.samples_per_s ? 'samples/s' : undefined,
              hint: data.kind === 'distributed' ? `${s.workers} 个 Worker 合计` : '单进程',
            },
            {
              label: '末 Epoch 耗时',
              value: fmtDuration(s.final?.epoch_train_wall_s),
              hint: s.epochs ? `共 ${s.epochs} 个 Epoch` : undefined,
            },
            { label: '总耗时', value: fmtDuration(elapsed), hint: live ? '仍在运行' : s.finished_unix_s ? `结束于 ${fmtTime(s.finished_unix_s)}` : undefined },
          ]}
        />
      </div>

      {s.status === 'pending' ? (
        <Alert style={{ marginBottom: 16 }} type="info" showIcon message="任务正在排队：单 GPU 同一时间只运行一个任务。" />
      ) : null}
      {s.error && !live ? (
        <Alert style={{ marginBottom: 16 }} type={s.status === 'cancelled' ? 'warning' : 'error'} showIcon message={s.error} />
      ) : null}

      <Tabs
        activeKey={activeTab}
        onChange={(k) => setSearchParams({ tab: k })}
        items={items}
        destroyOnHidden={false}
      />
    </div>
  )
}
