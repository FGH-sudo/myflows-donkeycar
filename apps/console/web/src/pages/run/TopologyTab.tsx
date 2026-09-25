import { Card, Col, Empty, Row, Table, Tag } from 'antd'
import type { EChartsOption } from 'echarts'
import { useMemo } from 'react'
import type { RankStatus, RunDetail } from '../../api'
import EChart from '../../components/EChart'
import { TRANSPORT_TEXT, fmtNum } from '../../format'

function syncColor(fraction: number | null | undefined) {
  if (fraction === null || fraction === undefined) return '#8c8c8c'
  if (fraction > 0.6) return '#f5222d'
  if (fraction > 0.35) return '#fa8c16'
  return '#52c41a'
}

export function topologyOption(mode: string, transport: string, workers: number, status: RankStatus[] = [], compact = false, device?: string | null): EChartsOption {
  const byRank = new Map(status.map((s) => [s.rank, s]))
  const radius = compact ? 90 : 170
  const nodes: Record<string, unknown>[] = []
  const links: Record<string, unknown>[] = []
  const label = (rank: number) => {
    const s = byRank.get(rank)
    if (!s || s.step === undefined || compact) return `rank ${rank}`
    const ms = s.step_wall_s ? `${(s.step_wall_s * 1000).toFixed(1)} ms` : '—'
    const pct = s.sync_fraction !== null && s.sync_fraction !== undefined ? `${(s.sync_fraction * 100).toFixed(0)}%` : '—'
    return `rank ${rank}\n已完成 ${s.steps_done} step\n${ms} · 同步 ${pct}`
  }
  const workerNode = (rank: number, x: number, y: number) => {
    const s = byRank.get(rank)
    const stale = s?.seconds_since_last !== null && s?.seconds_since_last !== undefined && s.seconds_since_last > 15
    nodes.push({
      id: `w${rank}`, name: `rank ${rank}`, x, y, symbolSize: compact ? 44 : 62,
      itemStyle: { color: stale ? '#bfbfbf' : syncColor(s?.sync_fraction), borderColor: '#fff', borderWidth: 2 },
      label: { show: true, formatter: label(rank), position: compact ? 'inside' : 'bottom', fontSize: compact ? 10 : 12, color: compact ? '#fff' : '#262626' },
    })
  }
  if (mode === 'ps') {
    nodes.push({ id: 'ps', name: 'PS', x: 0, y: 0, symbol: 'roundRect', symbolSize: compact ? [70, 34] : [110, 48],
      itemStyle: { color: '#1668dc' }, label: { show: true, formatter: compact ? 'PS' : 'PS（CPU 聚合）', color: '#fff', fontWeight: 'bold' } })
    for (let r = 0; r < workers; r += 1) {
      const angle = (2 * Math.PI * r) / workers - Math.PI / 2
      workerNode(r, Math.cos(angle) * radius, Math.sin(angle) * radius)
      links.push({ source: `w${r}`, target: 'ps', lineStyle: { width: 2, curveness: 0.12 }, label: { show: !compact && workers <= 4, formatter: 'Push 梯度', fontSize: 10 } })
      links.push({ source: 'ps', target: `w${r}`, lineStyle: { width: 2, curveness: 0.12, type: 'dashed' }, label: { show: !compact && workers <= 4, formatter: 'Pull 平均梯度', fontSize: 10 } })
    }
  } else if (mode === 'ring') {
    for (let r = 0; r < workers; r += 1) {
      const angle = (2 * Math.PI * r) / workers - Math.PI / 2
      workerNode(r, Math.cos(angle) * radius, Math.sin(angle) * radius)
    }
    for (let r = 0; r < workers && workers > 1; r += 1) {
      links.push({ source: `w${r}`, target: `w${(r + 1) % workers}`, lineStyle: { width: 3, curveness: 0.2, color: '#722ed1' } })
    }
  } else {
    workerNode(0, 0, 0)
  }
  return {
    animation: false,
    tooltip: { formatter: (p: unknown) => {
      const data = (p as { data?: { name?: string } }).data
      return data?.name ?? ''
    } },
    series: [{
      type: 'graph', layout: 'none', roam: !compact, data: nodes, links,
      edgeSymbol: ['none', 'arrow'], edgeSymbolSize: 9,
      lineStyle: { color: '#8c8c8c', opacity: 0.85 },
      emphasis: { focus: 'adjacency' },
      top: compact ? 24 : 72, bottom: compact ? 24 : 70,
    }],
    graphic: compact ? undefined : [{ type: 'text', left: 12, top: 10, style: { text: `${mode === 'ps' ? 'Parameter Server 星形' : mode === 'ring' ? 'Ring AllReduce 环形' : '单进程'} · ${TRANSPORT_TEXT[transport] ?? transport} · ${workers} Worker${device === 'cpu' ? '（CPU）' : '（共享单 GPU）'}`, fontSize: 13, fill: '#595959' } }],
  }
}

export default function TopologyTab({ detail, rankStatus, live }: { detail: RunDetail; rankStatus: RankStatus[]; live: boolean }) {
  const summary = detail.summary
  const option = useMemo(() => topologyOption(summary.mode ?? 'single', summary.transport ?? 'none', summary.workers, rankStatus, false, summary.device),
    [summary.mode, summary.transport, summary.workers, rankStatus, summary.device])
  if (detail.kind !== 'distributed') return <Card><Empty description="单进程训练没有分布式拓扑" /></Card>
  return (
    <Row gutter={[16, 16]}>
      <Col xs={24} xl={13}>
        <Card size="small" title="拓扑与各 rank 当前状态" extra={<span className="muted">颜色：梯度同步占整步比例 <Tag color="green">低</Tag><Tag color="orange">中</Tag><Tag color="red">高</Tag></span>}>
          <EChart option={option} height={460} />
        </Card>
      </Col>
      <Col xs={24} xl={11}>
        <Card size="small" title={live ? '各 rank 实时状态' : '各 rank 最终状态'}>
          <Table<RankStatus> size="small" rowKey="rank" pagination={false} dataSource={rankStatus}
            columns={[
              { title: 'rank', dataIndex: 'rank', width: 60 },
              { title: 'PID', dataIndex: 'pid', width: 80, render: (v) => v ?? '—' },
              { title: 'epoch', dataIndex: 'epoch', width: 64, render: (v) => (v === undefined ? '—' : v + 1) },
              { title: '已完成 step', dataIndex: 'steps_done', width: 96 },
              { title: 'loss', dataIndex: 'loss', render: (v) => fmtNum(v) },
              { title: 'step 耗时', dataIndex: 'step_wall_s', render: (v) => (v ? `${(v * 1000).toFixed(1)} ms` : '—') },
              { title: '同步占比', dataIndex: 'sync_fraction', render: (v) => (v === null || v === undefined ? '—' : <Tag color={syncColor(v) === '#52c41a' ? 'green' : syncColor(v) === '#fa8c16' ? 'orange' : 'red'}>{(v * 100).toFixed(0)}%</Tag>) },
              ...(live ? [{ title: '距上次更新', dataIndex: 'seconds_since_last', render: (v: number | null) => (v === null || v === undefined ? '—' : `${v.toFixed(1)} s`) }] : []),
            ]} />
        </Card>
      </Col>
    </Row>
  )
}
