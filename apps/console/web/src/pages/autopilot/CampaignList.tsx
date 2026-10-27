import { MessageOutlined, PlusOutlined, RightOutlined } from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { Button, Table, Tag, Tooltip } from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { Link, useNavigate } from 'react-router-dom'
import { autopilot } from '../../api'
import type { CampaignSummary } from '../../api'
import EmptyState from '../../components/EmptyState'
import { Panel } from '../../components/Panel'
import StatusTag from '../../components/StatusTag'
import { ACTIVE_CAMPAIGN, PHASE_TEXT, campaignPath, fmtDuration, fmtNum, fmtTime } from '../../format'

export default function CampaignList() {
  const navigate = useNavigate()
  const campaigns = useQuery({
    queryKey: ['autopilot', 'campaigns'],
    queryFn: autopilot.campaigns,
    refetchInterval: (query) => ((query.state.data ?? []).some((c) => ACTIVE_CAMPAIGN.includes(c.status)) ? 3000 : 15000),
  })

  const columns: ColumnsType<CampaignSummary> = [
    {
      title: 'Campaign',
      dataIndex: 'campaign_id',
      render: (cid: string, c) => (
        <div className="cell-stack">
          <Link to={campaignPath(cid)} className="table-link cell-title">
            {c.title || cid}
          </Link>
          <span className="cell-sub">
            {c.title ? `${cid} · ` : ''}
            {c.mode === 'formal' ? '正式' : '联调'}
            {c.created_by === 'llm' ? ' · LLM 创建' : ''}
          </span>
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      width: 150,
      render: (status: string, c) => (
        <div className="cell-stack">
          <StatusTag status={status} />
          <span className="cell-sub">阶段：{PHASE_TEXT[c.phase] ?? c.phase}</span>
        </div>
      ),
    },
    {
      title: '最佳',
      key: 'best',
      render: (_, c) =>
        c.best ? (
          <div className="cell-stack">
            <span className="num">{fmtNum(c.best.value, 5)}</span>
            <span className="cell-sub">
              {c.objective.metric} · {c.best.trial_id}
            </span>
          </div>
        ) : (
          <span className="muted">—</span>
        ),
    },
    {
      title: '预算用量',
      key: 'budget',
      render: (_, c) => (
        <div className="cell-stack">
          <span className="num">
            {c.used.trials_started} / {c.budget.max_trials} trial
          </span>
          <span className="cell-sub">
            {fmtDuration(c.used.run_s)} / {fmtDuration(c.budget.wall_time_s)}
          </span>
        </div>
      ),
    },
    {
      title: '待确认',
      dataIndex: 'pending',
      width: 90,
      render: (n: number) => (n ? <Tag color="warning">{n}</Tag> : <span className="muted">—</span>),
    },
    {
      title: 'LLM',
      key: 'llm',
      width: 110,
      render: (_, c) =>
        c.llm.enabled ? (
          <Tooltip title={c.llm.disabled_reason ?? '后台顾问调用次数'}>
            <span className={c.llm.disabled_reason ? 'muted' : ''}>{c.llm.calls} 次{c.llm.disabled_reason ? '（已关闭）' : ''}</span>
          </Tooltip>
        ) : (
          <span className="muted">未启用</span>
        ),
    },
    { title: '创建', dataIndex: 'created_unix_s', width: 140, render: (t: number) => <span className="muted">{fmtTime(t)}</span> },
    {
      key: 'go',
      width: 48,
      render: (_, c) => (
        <Button type="text" icon={<RightOutlined />} onClick={() => navigate(campaignPath(c.campaign_id))} aria-label="打开" />
      ),
    },
  ]

  return (
    <div>
      <div className="page-title">
        <div>
          <h2>AutoPilot</h2>
          <div className="page-subtitle">
            按 baseline → explore → exploit → confirm 自动调度训练、诊断瓶颈、复核最佳配置并生成报告；所有 trial 与手动任务共用一个 GPU 队列
          </div>
        </div>
        <div className="header-actions">
          <Button icon={<MessageOutlined />} onClick={() => navigate('/autopilot/assistant')}>
            AutoPilot 助手
          </Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => navigate('/autopilot/new')}>
            新建 campaign
          </Button>
        </div>
      </div>
      <Panel>
        <Table
          rowKey="campaign_id"
          columns={columns}
          dataSource={campaigns.data ?? []}
          loading={campaigns.isLoading}
          pagination={false}
          locale={{
            emptyText: (
              <EmptyState
                title="还没有 campaign"
                description="新建一个 campaign：先跑人工基线（lr 1e-3、batch 2、num_workers 0），再在 lr × batch 中抽样探索"
                action={
                  <Button type="primary" icon={<PlusOutlined />} onClick={() => navigate('/autopilot/new')}>
                    新建 campaign
                  </Button>
                }
              />
            ),
          }}
        />
      </Panel>
    </div>
  )
}
