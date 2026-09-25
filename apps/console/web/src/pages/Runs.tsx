import { ReloadOutlined } from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { Button, Card, Input, Select, Space, Table, Tag, Tooltip } from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { useMemo, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { api } from '../api'
import type { RunSummary } from '../api'
import StatusTag from '../components/StatusTag'
import { KIND_TEXT, fmtDuration, fmtNum, fmtTime, headlineMetric, isActive, modeLabel } from '../format'

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
    return (runs.data ?? []).filter((r) =>
      (!task || r.task === task) && (!mode || r.mode === mode) && (!status || r.status === status) &&
      (!needle || r.label.toLowerCase().includes(needle) || r.id.toLowerCase().includes(needle)))
  }, [runs.data, task, mode, status, q])

  const options = (key: keyof RunSummary) =>
    Array.from(new Set((runs.data ?? []).map((r) => r[key]).filter(Boolean) as string[])).sort().map((v) => ({ value: v, label: v }))

  const columns: ColumnsType<RunSummary> = [
    {
      title: '运行', dataIndex: 'label', width: 300, fixed: 'left',
      render: (label: string, r) => (
        <Space orientation="vertical" size={0}>
          <Tooltip title={r.id} placement="topLeft"><Link to={`/run/${r.id}`}>{label}</Link></Tooltip>
          <span className="muted" style={{ fontSize: 12 }}>{KIND_TEXT[r.kind] ?? r.kind}{r.device ? ` · ${r.device}` : ''}</span>
        </Space>
      ),
    },
    { title: '任务', dataIndex: 'task', width: 150, render: (v) => v ?? '—' },
    { title: '模式', width: 170, render: (_, r) => modeLabel(r) },
    { title: 'Worker', dataIndex: 'workers', width: 80, align: 'center', render: (v: number) => <Tag>{v}</Tag> },
    { title: '状态', dataIndex: 'status', width: 96, render: (s, r) => (r.error ? <Tooltip title={r.error}><span><StatusTag status={s} /></span></Tooltip> : <StatusTag status={s} />) },
    { title: '开始', dataIndex: 'started_unix_s', width: 130, render: fmtTime, sorter: (a, b) => (a.started_unix_s ?? 0) - (b.started_unix_s ?? 0), defaultSortOrder: 'descend' },
    { title: '耗时', dataIndex: 'elapsed_s', width: 100, render: (v, r) => (isActive(r.status) && r.started_unix_s ? fmtDuration(Date.now() / 1000 - r.started_unix_s) : fmtDuration(v)) },
    {
      title: '结果', width: 170,
      render: (_, r) => {
        const m = headlineMetric(r)
        return m ? <span><span className="muted">{m.label} </span>{m.value}</span> : '—'
      },
    },
    { title: '吞吐', width: 110, render: (_, r) => (r.final?.samples_per_s ? `${fmtNum(r.final.samples_per_s, 0)}/s` : '—') },
  ]

  return (
    <div>
      <div className="page-title">
        <h2>运行记录</h2>
        <Space>
          <Button disabled={selected.length < 2} onClick={() => navigate(`/compare?ids=${encodeURIComponent(selected.join(','))}`)}>
            对比所选（{selected.length}）
          </Button>
          <Button type="primary" onClick={() => navigate('/new')}>新建训练</Button>
        </Space>
      </div>
      <Card size="small" style={{ marginBottom: 16 }}>
        <Space wrap>
          <Select style={{ width: 220 }} loading={sources.isLoading} value={sources.data ? source : undefined} placeholder="来源" onChange={(v) => { setParams({ source: v }); setSelected([]) }}
            options={[{ value: 'all', label: '全部来源' }, ...(sources.data ?? []).map((s) => ({ value: s.key, label: s.label }))]} />
          <Select allowClear placeholder="任务" style={{ width: 170 }} value={task} onChange={setTask} options={options('task')} />
          <Select allowClear placeholder="模式" style={{ width: 120 }} value={mode} onChange={setMode} options={options('mode')} />
          <Select allowClear placeholder="状态" style={{ width: 120 }} value={status} onChange={setStatus} options={options('status')} />
          <Input.Search allowClear placeholder="按标签或路径搜索" style={{ width: 260 }} onSearch={setQ} onChange={(e) => !e.target.value && setQ('')} />
          <Button icon={<ReloadOutlined />} onClick={() => runs.refetch()} loading={runs.isFetching}>刷新</Button>
        </Space>
      </Card>
      <Card size="small">
        <Table<RunSummary> rowKey="id" size="middle" loading={runs.isLoading} columns={columns} dataSource={data}
          scroll={{ x: 1400 }}
          pagination={{ pageSize: 20, showSizeChanger: true, showTotal: (t) => `共 ${t} 条` }}
          rowSelection={{ selectedRowKeys: selected, onChange: (keys) => setSelected(keys as string[]), preserveSelectedRowKeys: true }}
          locale={{ emptyText: source === 'console' ? '还没有控制台任务，点击右上角“新建训练”' : '没有匹配的运行' }} />
      </Card>
    </div>
  )
}
