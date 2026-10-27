import {
  ArrowLeftOutlined,
  AuditOutlined,
  BulbOutlined,
  CaretRightOutlined,
  CheckOutlined,
  CloseOutlined,
  DashboardOutlined,
  FileTextOutlined,
  MessageOutlined,
  PauseOutlined,
  RobotOutlined,
  SettingOutlined,
  StopOutlined,
  TrophyOutlined,
} from '@ant-design/icons'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, App, Badge, Button, Popconfirm, Space, Spin, Table, Tabs, Tag, Tooltip } from 'antd'
import type { ColumnsType } from 'antd/es/table'
import type { ReactNode } from 'react'
import { useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { autopilot } from '../../api'
import type { CampaignDetail as Detail, CampaignEvent, DiagnosisChain, Finding, PendingItem, Trial } from '../../api'
import EmptyState from '../../components/EmptyState'
import { Panel, SectionHeader } from '../../components/Panel'
import StatStrip from '../../components/StatStrip'
import StatusTag from '../../components/StatusTag'
import { ACTIVE_CAMPAIGN, PHASE_TEXT, fmtDuration, fmtNum, fmtShare, fmtTime } from '../../format'
import { useEventSource } from '../../hooks/useEventSource'
import Markdown from './Markdown'

const PHASES = ['baseline', 'explore', 'exploit', 'confirm', 'done']

const tabLabel = (icon: ReactNode, text: string, count?: number) => (
  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
    {icon}
    {text}
    {count ? <Badge count={count} size="small" color="#c47a12" /> : null}
  </span>
)

const params = (p: Record<string, number>) =>
  Object.entries(p)
    .map(([k, v]) => `${k}=${typeof v === 'number' ? fmtNum(v, 6) : v}`)
    .join('、')

function PhaseTrack({ phase }: { phase: string }) {
  const at = PHASES.indexOf(phase)
  return (
    <div className="phase-track">
      {PHASES.map((p, i) => (
        <span key={p} className={`phase-step${i < at || phase === 'done' ? ' done' : i === at ? ' current' : ''}`}>
          {i < at || phase === 'done' ? <CheckOutlined /> : null}
          {PHASE_TEXT[p]}
        </span>
      ))}
    </div>
  )
}

const EVENT_TEXT: Record<string, (e: CampaignEvent) => string> = {
  campaign: (e) => `campaign ${String(e.status)}${e.reason ? `：${String(e.reason)}` : ''}${e.source ? `（${String(e.source)}）` : ''}`,
  phase: (e) => `进入阶段 ${PHASE_TEXT[String(e.phase)] ?? e.phase}${e.note ? `：${String(e.note)}` : ''}`,
  proposal: (e) => `建议（${String((e.origin as { strategy?: string })?.strategy ?? '')}）：${params(e.params as Record<string, number>)}`,
  precheck: (e) => (e.passed ? '预检通过' : `预检拒绝：${(e.reasons as string[]).join('；')}`),
  dispatch: (e) => `派发 ${String(e.trial_id)}（第 ${String(e.attempt)} 次尝试）`,
  verdict: (e) =>
    `${String(e.trial_id)} → ${String(e.status)}${e.verdict_kind ? `:${String(e.verdict_kind)}` : ''}${
      e.objective !== undefined && e.objective !== null ? ` · ${fmtNum(e.objective as number, 5)}` : ''
    }`,
  online_stop: (e) => `在线停止 ${String(e.trial_id)}：${String(e.message)}`,
  retry: (e) => `重试 ${String(e.trial_id)}（${String(e.retry_kind)}）`,
  skip: (e) => `跳过 ${String(e.trial_id)}：${String(e.reason)}`,
  finding: (e) => `[${String(e.agent)}] ${String(e.message)}`,
  pending: (e) => `待确认 ${String(e.pending_id)}（${String(e.pending_kind)}，来自 ${String(e.source)}）`,
  confirmation: (e) => `${String(e.pending_id)} ${e.status === 'confirmed' ? '已确认' : '已拒绝'}`,
  note: (e) => `${String(e.title ?? '说明')}${e.text ? `：${String(e.text).slice(0, 120)}` : ''}`,
  llm_call: (e) => {
    const usage = e.usage as { total_tokens?: number } | undefined
    const tools = ((e.tool_calls as { tool?: string; status?: string }[] | undefined) ?? [])
      .map((c) => `${String(c.tool ?? '').replace('mcp__autopilot__', '')}${c.status === 'error' ? '✗' : ''}`)
      .join(', ')
    const state = e.status === 'started' ? '开始' : e.status === 'ok' ? '完成' : `失败：${String(e.error ?? '')}`
    return `顾问调用（${String(e.reason ?? '')}）${state}${usage?.total_tokens ? ` · ${usage.total_tokens} token` : ''}${tools ? ` · ${tools}` : ''}`
  },
  llm_action: (e) =>
    `[${String(e.source ?? 'llm')}] ${String(e.tool)} ${String(e.args ?? '')}${e.ok === false ? ` → 被拒绝：${String(e.error ?? '')}` : ''}`,
  llm_error: (e) => `LLM 出错：${String(e.error)}`,
  llm_skip: (e) => `跳过顾问调用（${String(e.reason ?? '')}）：${String(e.why ?? '')}`,
  llm_disabled: (e) => `本 campaign 关闭 LLM：${String(e.why ?? '')}`,
  trial_stop: (e) => `停止 ${String(e.trial_id)}（${String(e.source)}）`,
  rollback: (e) =>
    `精确回滚 ${String(e.trial_id)}（${String(e.failure)}）：回到 attempt ${String(e.from_attempt)} 第 ${String(e.from_step)} 步，lr ${fmtNum(
      e.lr_before as number,
      3,
    )} → ${fmtNum(e.lr as number, 3)}`,
  rollback_skipped: (e) => `不回滚 ${String(e.trial_id)}（${String(e.failure)}）：${String(e.reason ?? '')}`,
  retry_skipped: (e) => `不重试 ${String(e.trial_id)}：${String(e.reason ?? '')}`,
  oom_derived: (e) =>
    e.derived
      ? `${String(e.trial_id)} 显存不足 → 派生 ${String(e.derived)}（${params(e.delta as Record<string, number>)}）`
      : `${String(e.trial_id)} 显存不足，派生的半 batch trial 未通过预检`,
  test_eval: (e) => {
    if (e.status === 'started') return `独立测试集评估开始（${String(e.items)} 个 checkpoint）`
    const groups = Object.entries((e.groups as Record<string, number | null>) ?? {})
      .map(([g, v]) => `${g === 'baseline' ? '基线' : '最佳候选'} ${v === null || v === undefined ? '—' : fmtNum(v, 5)}`)
      .join('，')
    return e.status === 'done' ? `独立测试集评估完成：${groups}` : '独立测试集评估失败'
  },
}

function EventList({ events }: { events: CampaignEvent[] }) {
  if (!events.length) return <EmptyState compact title="暂无事件" />
  return (
    <div style={{ maxHeight: 420, overflowY: 'auto' }}>
      {[...events].reverse().map((e) => (
        <div key={e.seq} style={{ display: 'flex', gap: 12, padding: '7px 20px', borderTop: '1px solid var(--border)' }}>
          <span className="muted mono" style={{ flexShrink: 0 }}>
            {fmtTime(e.unix_s)}
          </span>
          <Tag style={{ flexShrink: 0 }}>{e.kind}</Tag>
          <span style={{ minWidth: 0 }}>{(EVENT_TEXT[e.kind] ?? ((x) => JSON.stringify(x).slice(0, 160)))(e)}</span>
        </div>
      ))}
    </div>
  )
}

function Overview({ c, events }: { c: Detail; events: CampaignEvent[] }) {
  const noise = c.state.noise
  return (
    <div style={{ display: 'grid', gap: 'calc(24px * var(--s))' }}>
      <StatStrip
        items={[
          {
            label: `最佳 ${c.objective.metric}`,
            value: c.best ? fmtNum(c.best.value, 5) : '—',
            hint: c.best ? `${c.best.trial_id} · ${params(c.best.params)}` : '还没有有效 trial',
          },
          { label: 'trial', value: `${c.used.trials_started} / ${c.budget.max_trials}`, hint: `有效 ${c.valid} 个 · 尝试 ${c.used.attempts} 次` },
          { label: '运行时长', value: fmtDuration(c.used.run_s), hint: `预算 ${fmtDuration(c.budget.wall_time_s)}（不含排队）` },
          {
            label: '噪声水平',
            value: noise ? fmtNum(noise.std, 4) : '—',
            hint: noise ? `基线 ${noise.seeds.length} 个 seed：${fmtNum(noise.mean, 5)} ± ${fmtNum(noise.std, 4)}` : '基线完成后计算',
          },
          {
            label: 'LLM 调用',
            value: c.spec.llm.enabled ? `${c.state.llm.calls} / ${c.spec.llm.max_calls}` : '未启用',
            hint: c.state.llm.disabled_reason ?? (c.spec.llm.enabled
              ? `${c.state.llm.tokens.toLocaleString()}${c.spec.llm.max_total_tokens ? ` / ${c.spec.llm.max_total_tokens.toLocaleString()}` : ''} token`
              : '规则策略'),
          },
        ]}
      />
      <div>
        <SectionHeader title="阶段" description={c.state.phase_log.length ? c.state.phase_log[c.state.phase_log.length - 1].note : undefined} />
        <PhaseTrack phase={c.phase} />
        {Object.keys(c.state.adopted ?? {}).length ? (
          <div className="muted" style={{ marginTop: 10 }}>
            吞吐参数已沿用：{params(c.state.adopted)}
          </div>
        ) : null}
      </div>
      <div>
        <SectionHeader title="事件" description="每次建议、预检、派发、在线停止、Verdict 与 LLM 操作都会记录" />
        <Panel>
          <EventList events={events.slice(-200)} />
        </Panel>
      </div>
    </div>
  )
}

function Leaderboard({ c, trials, onStop }: { c: Detail; trials: Trial[]; onStop: (tid: string) => void }) {
  const columns: ColumnsType<Trial> = [
    { title: '名次', dataIndex: 'rank', width: 64, render: (r: number | null) => (r ? <span className="num">{r}</span> : <span className="muted">—</span>) },
    {
      title: 'trial',
      dataIndex: 'trial_id',
      render: (tid: string, t) => {
        const job = t.attempts[t.attempts.length - 1]?.job_id
        return (
          <div className="cell-stack">
            {job ? (
              <Link className="table-link cell-title" to={`/run/console/${job}`}>
                {tid}
              </Link>
            ) : (
              <span className="cell-title">{tid}</span>
            )}
            <span className="cell-sub">
              {PHASE_TEXT[t.phase]} · {t.origin.strategy}
              {t.origin.source === 'llm' || t.origin.source?.startsWith('llm') ? ' · LLM' : ''}
            </span>
          </div>
        )
      },
    },
    { title: '参数', dataIndex: 'params', render: (p: Record<string, number>) => <span className="mono">{params(p)}</span> },
    {
      title: c.objective.metric,
      dataIndex: 'objective',
      sorter: (a, b) => (a.objective ?? Infinity) - (b.objective ?? Infinity),
      render: (v: number | null) => <span className="num">{fmtNum(v, 5)}</span>,
    },
    { title: 'samples/s', key: 'tp', render: (_, t) => <span className="num">{fmtNum(t.result?.summary.throughput_sps, 1)}</span> },
    { title: '数据读取', key: 'ds', render: (_, t) => fmtShare(t.result?.summary.data_share) },
    { title: 'GPU 利用率', key: 'gu', render: (_, t) => (t.result?.summary.gpu_util_median != null ? `${fmtNum(t.result.summary.gpu_util_median, 0)}%` : '—') },
    {
      title: 'Verdict',
      key: 'verdict',
      render: (_, t) => {
        const reasons = t.verdict?.reasons?.filter(Boolean) ?? []
        const label = t.verdict ? t.verdict.status : t.status
        return (
          <Tooltip title={reasons.join('；') || t.skip_reason}>
            <Space size={4}>
              <StatusTag status={label} />
              {t.verdict?.kind ? <Tag>{t.verdict.kind}</Tag> : null}
              {t.attempts.filter((a) => a.cause === 'rollback').length ? (
                <Tag color="purple">回滚 {t.attempts.filter((a) => a.cause === 'rollback').length}</Tag>
              ) : null}
              {t.attempts.filter((a) => a.cause !== 'rollback').length > 1 ? (
                <Tag>重试 {t.attempts.filter((a) => a.cause !== 'rollback').length - 1}</Tag>
              ) : null}
            </Space>
          </Tooltip>
        )
      },
    },
    {
      key: 'act',
      width: 64,
      render: (_, t) =>
        t.status === 'running' || t.status === 'queued' ? (
          <Popconfirm title={`停止 ${t.trial_id}？`} onConfirm={() => onStop(t.trial_id)} okButtonProps={{ danger: true }}>
            <Button size="small" danger type="text" icon={<StopOutlined />} aria-label="停止" />
          </Popconfirm>
        ) : null,
    },
  ]
  return (
    <Panel>
      <Table rowKey="trial_id" columns={columns} dataSource={trials} pagination={false} size="middle" scroll={{ x: true }} />
    </Panel>
  )
}

function ChainCard({ chain, findings }: { chain: DiagnosisChain; findings: Record<string, Finding> }) {
  const rows = [
    ['目标指标', fmtNum(chain.before.objective, 5), fmtNum(chain.after.objective, 5)],
    ['samples/s（含数据读取）', fmtNum(chain.before.throughput_sps, 1), fmtNum(chain.after.throughput_sps, 1)],
    ['数据读取占比', fmtShare(chain.before.data_share), fmtShare(chain.after.data_share)],
    ['GPU 利用率中位数', fmtNum(chain.before.gpu_util_median, 0), fmtNum(chain.after.gpu_util_median, 0)],
  ]
  const color = chain.decision === 'adopted' || chain.decision === 'better' || chain.decision === 'recovered' ? 'success' : chain.decision === 'failed' ? 'error' : 'default'
  return (
    <div className="chain">
      <div className="chain-head">
        <Tag color="blue">{chain.proposal.agent}</Tag>
        <span>
          {chain.base_trial} → {chain.trial_id}
        </span>
        <Tag color={color}>{chain.decision}</Tag>
      </div>
      <ol className="chain-steps">
        {chain.finding_ids.map((fid) => (
          <li key={fid}>
            发现：{findings[fid]?.message ?? fid}
            {(findings[fid]?.evidence ?? []).map((e, i) => (
              <div key={i} className="chain-evidence">
                证据：{e.metric} = {fmtNum(e.value, 4)}
                {e.threshold !== undefined ? `（阈值 ${e.threshold}）` : ''} · <span className="mono">{e.file}</span>
              </div>
            ))}
          </li>
        ))}
        <li>建议：{chain.proposal.rationale}（{params(chain.proposal.delta)}）</li>
        <li>派生 trial {chain.trial_id}，与 {chain.base_trial} 同 seed 配对</li>
        <li>结论：{chain.reason}</li>
      </ol>
      <table className="markdown-body" style={{ display: 'table' }}>
        <thead>
          <tr>
            <th />
            <th>基准 {chain.base_trial}</th>
            <th>派生 {chain.trial_id}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([k, a, b]) => (
            <tr key={k}>
              <td>{k}</td>
              <td className="num">{a}</td>
              <td className="num">{b}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Diagnosis({ cid }: { cid: string }) {
  const data = useQuery({ queryKey: ['autopilot', cid, 'findings'], queryFn: () => autopilot.findings(cid) })
  if (!data.data) return <Spin />
  const byId = Object.fromEntries(data.data.findings.map((f) => [f.id, f]))
  const others = data.data.findings.filter((f) => f.severity !== 'action')
  return (
    <div style={{ display: 'grid', gap: 'calc(24px * var(--s))' }}>
      <div>
        <SectionHeader title="诊断链" description="发现 → 建议 → 派生 trial → 前后对照 → 决定" />
        <Panel>
          {data.data.chains.length ? (
            data.data.chains.map((chain) => <ChainCard key={chain.trial_id} chain={chain} findings={byId} />)
          ) : (
            <EmptyState compact title="还没有由诊断派生的 trial" description="GPU Monitor 等子 Agent 在 exploit 阶段根据证据提出建议" />
          )}
        </Panel>
      </div>
      <div>
        <SectionHeader title="其他发现" />
        <Panel>
          {others.length ? (
            <Table
              rowKey="id"
              size="small"
              pagination={false}
              dataSource={others}
              columns={[
                { title: '来源', dataIndex: 'agent', width: 130 },
                { title: '类型', dataIndex: 'kind', width: 170, render: (k: string) => <Tag>{k}</Tag> },
                { title: 'trial', dataIndex: 'trial_id', width: 80 },
                { title: '内容', dataIndex: 'message' },
              ]}
            />
          ) : (
            <EmptyState compact title="暂无" />
          )}
        </Panel>
      </div>
      {Object.keys(data.data.unavailable ?? {}).length ? (
        <Alert
          type="info"
          showIcon
          message="不可用的子 Agent"
          description={Object.entries(data.data.unavailable).map(([k, v]) => (
            <div key={k}>
              {k}：{v}
            </div>
          ))}
        />
      ) : null}
    </div>
  )
}

function Pending({ cid, items, onResolve }: { cid: string; items: PendingItem[]; onResolve: (pid: string, ok: boolean) => void }) {
  if (!items.length)
    return (
      <Panel>
        <EmptyState title="没有待确认项" description="LLM 发起、会启动训练的操作在这里等待人工确认；确认前不会运行" />
      </Panel>
    )
  return (
    <Panel>
      {[...items].reverse().map((item) => (
        <div className="pending-card" key={`${cid}-${item.id}`}>
          <div className="chain-head">
            <span className="mono">{item.id}</span>
            <Tag>{item.kind === 'trials' ? '提交 trial' : '启动 campaign'}</Tag>
            <span className="muted">来自 {item.source} · {fmtTime(item.created_unix_s)}</span>
            <StatusTag status={item.status} />
          </div>
          {item.proposals.map((p, i) => (
            <div key={i} style={{ marginBottom: 8 }}>
              <div>
                <span className="mono">{params(p.delta)}</span>
                {p.base_trial ? <span className="muted">（基于 {p.base_trial}）</span> : null}
                {p.precheck.passed ? <Tag color="success" style={{ marginLeft: 8 }}>预检通过</Tag> : <Tag color="error" style={{ marginLeft: 8 }}>预检未通过</Tag>}
              </div>
              {p.rationale ? <div className="muted">理由（LLM 生成）：{p.rationale}</div> : null}
              {p.expected ? <div className="muted">预期：{p.expected}</div> : null}
              {p.precheck.reasons.length ? <div style={{ color: 'var(--red)' }}>{p.precheck.reasons.join('；')}</div> : null}
            </div>
          ))}
          {item.status === 'pending_confirmation' ? (
            <Space>
              <Button type="primary" icon={<CheckOutlined />} onClick={() => onResolve(item.id, true)} disabled={item.kind === 'trials' && !item.precheck.passed}>
                确认运行
              </Button>
              <Button icon={<CloseOutlined />} onClick={() => onResolve(item.id, false)}>
                拒绝
              </Button>
            </Space>
          ) : null}
        </div>
      ))}
    </Panel>
  )
}

function Report({ cid, available }: { cid: string; available: boolean }) {
  const { message } = App.useApp()
  const queryClient = useQueryClient()
  const [regenerating, setRegenerating] = useState(false)
  const report = useQuery({ queryKey: ['autopilot', cid, 'report', available], queryFn: () => autopilot.report(cid), enabled: available, retry: false })
  if (!available) return <Panel><EmptyState title="报告尚未生成" description="campaign 结束（完成、停止或预算用完）时自动生成 report.md、leaderboard.csv 与图表" /></Panel>
  if (!report.data) return <Spin />
  const regenerate = async () => {
    setRegenerating(true)
    try {
      await autopilot.regenerateReport(cid)
      await queryClient.invalidateQueries({ queryKey: ['autopilot', cid] })
      message.success('已按当前规则重新生成报告')
    } catch (e) {
      message.error((e as Error).message)
    } finally {
      setRegenerating(false)
    }
  }
  return (
    <Panel padded>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
        <div className="muted mono" style={{ flex: 1, minWidth: 0, overflowWrap: 'anywhere' }}>
          {report.data.path}
        </div>
        <Tooltip title="按当前规则重新诊断已记录的证据并重写报告；不会启动新的 trial">
          <Button size="small" loading={regenerating} onClick={regenerate}>
            重新生成
          </Button>
        </Tooltip>
      </div>
      <Markdown text={report.data.markdown} resolveImage={(src) => autopilot.figureUrl(cid, src.replace(/^figures\//, ''))} />
    </Panel>
  )
}

export default function CampaignDetail() {
  const { cid = '' } = useParams()
  const navigate = useNavigate()
  const [search, setSearch] = useSearchParams()
  const tab = search.get('tab') || 'overview'
  const { message } = App.useApp()
  const queryClient = useQueryClient()
  const campaign = useQuery({
    queryKey: ['autopilot', cid, 'detail'],
    queryFn: () => autopilot.campaign(cid),
    refetchInterval: (q) => (q.state.data && ACTIVE_CAMPAIGN.includes(q.state.data.status) ? 4000 : false),
  })
  const active = !!campaign.data && ACTIVE_CAMPAIGN.includes(campaign.data.status)
  const trials = useQuery({ queryKey: ['autopilot', cid, 'trials'], queryFn: () => autopilot.trials(cid), refetchInterval: active ? 4000 : false })
  const events = useQuery({ queryKey: ['autopilot', cid, 'events'], queryFn: () => autopilot.events(cid), refetchInterval: active ? 4000 : false })
  const refresh = () => queryClient.invalidateQueries({ queryKey: ['autopilot', cid] })
  useEventSource(active ? autopilot.streamUrl(cid) : null, { event: refresh, summary: refresh, end: refresh })

  if (campaign.isLoading) return <Spin style={{ display: 'block', marginTop: 80 }} />
  if (!campaign.data) return <Alert type="error" showIcon message="无法加载 campaign" description={String((campaign.error as Error)?.message ?? '')} />
  const c = campaign.data

  const act = async (action: 'start' | 'pause' | 'resume' | 'stop') => {
    try {
      await autopilot.action(cid, action)
      refresh()
    } catch (e) {
      message.error((e as Error).message)
    }
  }
  const resolve = async (pid: string, ok: boolean) => {
    try {
      await autopilot.resolve(cid, pid, ok)
      message.success(ok ? '已确认' : '已拒绝')
      refresh()
    } catch (e) {
      message.error((e as Error).message)
    }
  }
  const stopTrial = async (tid: string) => {
    try {
      await autopilot.stopTrial(cid, tid)
      refresh()
    } catch (e) {
      message.error((e as Error).message)
    }
  }
  const llmEvents = (events.data ?? []).filter(
    (e) => e.kind.startsWith('llm') || (typeof e.source === 'string' && e.source.startsWith('llm')),
  )
  const pending = c.state.pending ?? []
  const waiting = pending.filter((p) => p.status === 'pending_confirmation').length

  return (
    <div>
      <div className="run-header">
        <div className="run-header-main">
          <Tooltip title="返回">
            <Button type="text" icon={<ArrowLeftOutlined />} onClick={() => navigate('/autopilot')} aria-label="返回" className="run-back" />
          </Tooltip>
          <div style={{ minWidth: 0 }}>
            <div className="run-title-row">
              <h2 className="run-title">{c.title || c.campaign_id}</h2>
              <StatusTag status={c.status} />
            </div>
            <div className="run-meta">
              <span>{c.campaign_id}</span>
              <span>{c.mode === 'formal' ? '正式' : '联调'}</span>
              <span>阶段：{PHASE_TEXT[c.phase]}</span>
              <span>目标：{c.objective.metric}</span>
              {c.created_by === 'llm' ? <span>LLM 创建</span> : null}
              {c.started_unix_s ? <span>开始于 {fmtTime(c.started_unix_s)}</span> : null}
            </div>
          </div>
        </div>
        <Space wrap>
          <Button icon={<MessageOutlined />} onClick={() => navigate(`/autopilot/assistant?campaign=${encodeURIComponent(cid)}`)}>
            助手
          </Button>
          {c.status === 'draft' || c.status === 'paused' ? (
            <Button type="primary" icon={<CaretRightOutlined />} onClick={() => act(c.status === 'draft' ? 'start' : 'resume')}>
              {c.status === 'draft' ? '启动' : '继续'}
            </Button>
          ) : null}
          {c.status === 'running' ? (
            <Button icon={<PauseOutlined />} onClick={() => act('pause')}>
              暂停
            </Button>
          ) : null}
          {['draft', 'paused', 'running', 'pausing'].includes(c.status) ? (
            <Popconfirm title="停止 campaign？" description="正在运行的 trial 会先保存断点再停止，然后生成报告。" onConfirm={() => act('stop')} okButtonProps={{ danger: true }}>
              <Button danger icon={<StopOutlined />}>
                停止
              </Button>
            </Popconfirm>
          ) : null}
        </Space>
      </div>
      {c.stop_reason && !active ? <Alert style={{ marginBottom: 16 }} type={c.status === 'failed' ? 'error' : 'info'} showIcon message={`结束原因：${c.stop_reason}`} /> : null}
      {waiting ? (
        <Alert
          style={{ marginBottom: 16 }}
          type="warning"
          showIcon
          message={`${waiting} 个 LLM 发起的启动训练请求等待确认`}
          action={<Button size="small" onClick={() => setSearch({ tab: 'pending' })}>查看</Button>}
        />
      ) : null}
      <Tabs
        activeKey={tab}
        onChange={(k) => setSearch({ tab: k })}
        items={[
          { key: 'overview', label: tabLabel(<DashboardOutlined />, '概览'), children: <Overview c={c} events={events.data ?? []} /> },
          { key: 'trials', label: tabLabel(<TrophyOutlined />, '排行榜'), children: <Leaderboard c={c} trials={trials.data ?? []} onStop={stopTrial} /> },
          { key: 'diagnosis', label: tabLabel(<BulbOutlined />, '诊断链'), children: <Diagnosis cid={cid} /> },
          { key: 'pending', label: tabLabel(<AuditOutlined />, '待确认', waiting), children: <Pending cid={cid} items={pending} onResolve={resolve} /> },
          {
            key: 'llm',
            label: tabLabel(<RobotOutlined />, 'LLM 操作记录'),
            children: (
              <Panel>
                <EventList events={llmEvents} />
              </Panel>
            ),
          },
          { key: 'report', label: tabLabel(<FileTextOutlined />, '报告'), children: <Report cid={cid} available={c.report_available} /> },
          {
            key: 'config',
            label: tabLabel(<SettingOutlined />, '配置'),
            children: (
              <Panel padded>
                <pre className="json-view">{JSON.stringify({ spec: c.spec, provenance: c.state.provenance }, null, 2)}</pre>
              </Panel>
            ),
          },
        ]}
      />
    </div>
  )
}
