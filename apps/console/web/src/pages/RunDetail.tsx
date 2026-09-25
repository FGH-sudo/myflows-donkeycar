import { ArrowLeftOutlined, StopOutlined } from '@ant-design/icons'
import { useQueryClient } from '@tanstack/react-query'
import { App, Alert, Button, Popconfirm, Space, Spin, Tabs, Tag } from 'antd'
import { useParams, useNavigate } from 'react-router-dom'
import { api } from '../api'
import StatusTag from '../components/StatusTag'
import { KIND_TEXT, fmtDuration, modeLabel } from '../format'
import CurvesTab from './run/CurvesTab'
import LogTab from './run/LogTab'
import ResourcesTab from './run/ResourcesTab'
import SummaryTab from './run/SummaryTab'
import TimingTab from './run/TimingTab'
import TopologyTab from './run/TopologyTab'
import { useRunData } from './run/useRunData'

export default function RunDetailPage() {
  const params = useParams()
  const runId = params['*'] ?? ''
  const navigate = useNavigate()
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
    { key: 'curves', label: '训练曲线', children: <CurvesTab detail={data} state={state} ranks={ranks} /> },
    { key: 'timing', label: '耗时拆分', children: <TimingTab detail={data} state={state} ranks={ranks} /> },
    { key: 'resources', label: '资源监控', children: <ResourcesTab detail={data} state={state} /> },
    ...(data.kind === 'distributed' ? [{ key: 'topology', label: '分布式拓扑', children: <TopologyTab detail={data} rankStatus={rankStatus} live={live} /> }] : []),
    { key: 'summary', label: '配置与结果', children: <SummaryTab detail={data} /> },
    { key: 'log', label: '日志', children: <LogTab runId={runId} live={live} /> },
  ]

  return (
    <div>
      <div className="page-title">
        <Space size={12} wrap>
          <Button icon={<ArrowLeftOutlined />} onClick={() => navigate(-1)} />
          <h2>{s.label}</h2>
          <StatusTag status={s.status} />
          <Tag>{KIND_TEXT[s.kind] ?? s.kind}</Tag>
          <Tag color="blue">{modeLabel(s)}</Tag>
          {data.kind === 'distributed' ? <Tag color="purple">{s.workers} Worker</Tag> : null}
          <span className="muted">{fmtDuration(elapsed)}</span>
          {live ? <span className="muted">{connected ? <><span className="live-dot" />实时</> : '连接中…'}</span> : null}
        </Space>
        {live && jobId ? (
          <Popconfirm title="停止该任务？" description="分布式任务会结束全部进程；单进程任务会先保存断点再退出。" onConfirm={stop} okText="停止" okButtonProps={{ danger: true }}>
            <Button danger icon={<StopOutlined />} disabled={s.status === 'stopping'}>{s.status === 'stopping' ? '停止中' : '停止任务'}</Button>
          </Popconfirm>
        ) : null}
      </div>
      {s.status === 'pending' ? <Alert style={{ marginBottom: 16 }} type="info" showIcon message="任务正在排队：单 GPU 同一时间只运行一个任务。" /> : null}
      {s.error && !live ? <Alert style={{ marginBottom: 16 }} type={s.status === 'cancelled' ? 'warning' : 'error'} showIcon message={s.error} /> : null}
      <Tabs items={items} destroyOnHidden={false} />
    </div>
  )
}
