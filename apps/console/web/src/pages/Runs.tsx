import { LineChartOutlined, ReloadOutlined, RightOutlined, SearchOutlined } from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { Button, Input, Select, Table, Tooltip } from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { useMemo, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { api } from '../api'
import type { RunSummary } from '../api'
import EmptyState from '../components/EmptyState'
import { Panel } from '../components/Panel'
import StatusTag from '../components/StatusTag'
import { KIND_TEXT, MODE_TEXT, STATUS_META, fmtDuration, fmtNum, fmtTime, headlineMetric, isActive, modeLabel } from '../format'

export default function Runs() {
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()
  const source = params.get('source') ?? 'console'
  const [task, setTask] = useState<string | undefined>()
  const [mode, setMode] = useState<string | undefined>()
  const [status, setStatus] = useState<string | undefined>()
  const [q, setQ] = useState('')
  const [selected, setSelected] = useState<string[]>([])

  const sources = useQuery({ queryKey: ['sources'], queryFn: api.sources })
  const runs = useQuery({
    queryKey: ['runs', source],
    queryFn: () => api.runs({ source: source === 'all' ? undefined : source }),
    refetchInterval: (query) => ((query.state.data ?? []).some((r) => isActive(r.status)) ? 3000 : 15000),
  })

  const data = useMemo(() => {
    const needle = q.trim().toLowerCase()
    return (runs.data ?? []).filter(
      (r) =>
        (!task || r.task === task) &&
        (!mode || r.mode === mode) &&
        (!status || r.status === status) &&
        (!needle || r.label.toLowerCase().includes(needle) || r.id.toLowerCase().includes(needle)),
    )
  }, [runs.data, task, mode, status, q])

  const options = (key: keyof RunSummary, text?: Record<string, { label: string } | string>) =>
    Array.from(new Set((runs.data ?? []).map((r) => r[key]).filter(Boolean) as string[]))
      .sort()
      .map((v) => {
        const t = text?.[v]
        return { value: v, label: typeof t === 'string' ? t : (t?.label ?? v) }
      })

  const columns: ColumnsType<RunSummary> = [
    {
      title: '运行',
      dataIndex: 'label',
      render: (label: string, r) => (
        <div className="cell-stack">
          <Tooltip title={r.id} placement="topLeft">
            <Link to={`/run/${r.id}`} className="table-link cell-title">
              {label}
            </Link>
          </Tooltip>
          <span className="cell-sub">
            {[r.task, KIND_TEXT[r.kind] ?? r.kind, r.device].filter(Boolean).join(' · ')}
          </span>
        </div>
      ),
    },
    {
      title: '架构',
      width: 150,
      render: (_, r) => (
        <div className="cell-stack">
          <span className="nowrap">{modeLabel(r)}</span>
          <span className="cell-sub">{r.workers} Worker</span>
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      width: 96,
      render: (s, r) =>
        r.error ? (
          <Tooltip title={r.error}>
            <span>
              <StatusTag status={s} />
            </span>
          </Tooltip>
        ) : (
          <StatusTag status={s} />
        ),
    },
    {
      title: '开始时间',
      dataIndex: 'started_unix_s',
      width: 140,
      render: (v) => <span className="num muted">{fmtTime(v)}</span>,
      sorter: (a, b) => (a.started_unix_s ?? 0) - (b.started_unix_s ?? 0),
      defaultSortOrder: 'descend',
    },
    {
      title: '耗时',
      dataIndex: 'elapsed_s',
      width: 100,
      align: 'right',
      render: (v, r) => (
        <span className="num">
          {isActive(r.status) && r.started_unix_s ? fmtDuration(Date.now() / 1000 - r.started_unix_s) : fmtDuration(v)}
        </span>
      ),
    },
    {
      title: '核心指标',
      width: 130,
      align: 'right',
      render: (_, r) => {
        const m = headlineMetric(r)
        return m ? (
          <div className="cell-stack" style={{ alignItems: 'flex-end' }}>
            <span className="num" style={{ fontWeight: 600 }}>{m.value}</span>
            <span className="cell-sub">{m.label}</span>
          </div>
        ) : (
          <span className="muted">—</span>
        )
      },
    },
    {
      title: '吞吐',
      width: 110,
      align: 'right',
      render: (_, r) =>
        r.final?.samples_per_s ? (
          <span className="num">
            <span style={{ fontWeight: 600 }}>{fmtNum(r.final.samples_per_s, 0)}</span>
            <span className="muted"> /s</span>
          </span>
        ) : (
          <span className="muted">—</span>
        ),
    },
    {
      title: '',
      width: 80,
      align: 'right',
      render: (_, r) => (
        <Link to={`/run/${r.id}`} className="row-action">
          详情
          <RightOutlined />
        </Link>
      ),
    },
  ]

  return (
    <div>
      <div className="page-title">
        <div>
          <h2>运行记录</h2>
          <div className="page-subtitle">检索单进程与分布式训练历史，勾选多条后可横向对比</div>
        </div>
      </div>

      <Panel>
        <div className="table-toolbar">
          <div className="table-toolbar-filters">
            <Select
              style={{ width: 168 }}
              loading={sources.isLoading}
              value={sources.data ? source : undefined}
              placeholder="数据源"
              popupMatchSelectWidth={false}
              onChange={(v) => {
                setParams({ source: v })
                setSelected([])
              }}
              options={[{ value: 'all', label: '全部数据源' }, ...(sources.data ?? []).map((s) => ({ value: s.key, label: s.label }))]}
            />
            <Select allowClear placeholder="任务" style={{ width: 160 }} value={task} onChange={setTask} options={options('task')} popupMatchSelectWidth={false} />
            <Select allowClear placeholder="架构" style={{ width: 120 }} value={mode} onChange={setMode} options={options('mode', MODE_TEXT)} />
            <Select allowClear placeholder="状态" style={{ width: 120 }} value={status} onChange={setStatus} options={options('status', STATUS_META)} />
            <Input
              allowClear
              prefix={<SearchOutlined style={{ color: 'var(--text-faint)' }} />}
              placeholder="搜索名称或 ID"
              style={{ width: 220 }}
              value={q}
              onChange={(e) => setQ(e.target.value)}
            />
          </div>
          <span className="toolbar-item">
            <span className="num">{data.length} 条</span>
            <Tooltip title="刷新">
              <Button type="text" icon={<ReloadOutlined />} onClick={() => runs.refetch()} loading={runs.isFetching} aria-label="刷新" />
            </Tooltip>
          </span>
        </div>
        <Table<RunSummary>
          rowKey="id"
          size="middle"
          loading={runs.isLoading}
          columns={columns}
          dataSource={data}
          scroll={{ x: 1080 }}
          pagination={{
            pageSize: 20,
            showSizeChanger: true,
            showTotal: (t) => `共 ${t} 条`,
            style: { padding: '0 16px', margin: '14px 0' },
          }}
          rowSelection={{
            selectedRowKeys: selected,
            onChange: (keys) => setSelected(keys as string[]),
            preserveSelectedRowKeys: true,
            columnWidth: 48,
          }}
          locale={{
            emptyText: (
              <EmptyState
                title={source === 'console' ? '还没有从控制台发起的任务' : '没有匹配的运行记录'}
                description={source === 'console' ? '点击右上角“新建训练”发起第一个任务' : '试试调整筛选条件或切换数据源'}
              />
            ),
          }}
        />
      </Panel>

      {selected.length ? (
        <div className="selection-bar" role="region" aria-label="已选运行">
          <span className="num">已选 {selected.length} 条</span>
          <span className="selection-bar-sep" />
          <Button type="text" size="small" onClick={() => setSelected([])} className="selection-bar-ghost">
            清除
          </Button>
          <Button
            size="small"
            icon={<LineChartOutlined />}
            disabled={selected.length < 2}
            onClick={() => navigate(`/compare?ids=${encodeURIComponent(selected.join(','))}`)}
          >
            {selected.length < 2 ? '再选 1 条即可对比' : '对比所选'}
          </Button>
        </div>
      ) : null}
    </div>
  )
}
