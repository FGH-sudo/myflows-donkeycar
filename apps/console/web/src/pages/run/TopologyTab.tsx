import { Col, Row, Table } from 'antd'
import type { EChartsOption } from 'echarts'
import { useMemo } from 'react'
import type { RankStatus, RunDetail } from '../../api'
import EChart from '../../components/EChart'
import EmptyState from '../../components/EmptyState'
import { Panel } from '../../components/Panel'
import { TRANSPORT_TEXT, fmtNum, rankColor } from '../../format'

function syncColor(fraction: number | null | undefined) {
  if (fraction === null || fraction === undefined) return '#a1a1aa'
  if (fraction > 0.6) return '#d14343'
  if (fraction > 0.35) return '#c47a12'
  return '#2f9e62'
}

const SYNC_LEVELS: [string, string][] = [
  ['#2f9e62', '低 <35%'],
  ['#c47a12', '中 35–60%'],
  ['#d14343', '高 >60%'],
]

const EDGE = { push: '#3b6fd8', pull: '#2f9e7a', ring: '#8866d6' }

function EdgeKey({ color, dashed, text }: { color: string; dashed?: boolean; text: string }) {
  return (
    <span className="toolbar-item" style={{ gap: 6 }}>
      <svg width="18" height="2" aria-hidden>
        <line x1="0" y1="1" x2="18" y2="1" stroke={color} strokeWidth="2" strokeDasharray={dashed ? '4 3' : undefined} />
      </svg>
      {text}
    </span>
  )
}

function SyncValue({ value }: { value: number }) {
  return (
    <span className="num" style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontWeight: 600 }}>
      <span className="status-dot" style={{ background: syncColor(value) }} />
      {(value * 100).toFixed(0)}%
    </span>
  )
}

export function topologyOption(
  mode: string,
  _transport: string,
  workers: number,
  status: RankStatus[] = [],
  compact = false,
  _device?: string | null,
): EChartsOption {
  const byRank = new Map(status.map((s) => [s.rank, s]))
  const radius = compact ? 85 : mode === 'ps' ? 150 : 120
  const nodes: Record<string, unknown>[] = []
  const links: Record<string, unknown>[] = []
  // layout 'none' stretches the node bounding box to the series box on each
  // axis independently. Invisible corner anchors pin that box to roughly the
  // card's aspect ratio so the ring stays round and outer labels have room.
  const [halfX, halfY] = compact ? [radius + 110, radius + 24] : [radius + 190, radius + 44]
  for (const [x, y] of [[-halfX, -halfY], [halfX, halfY]]) {
    nodes.push({ id: `anchor${x}`, name: '', x, y, symbolSize: 0, label: { show: false }, tooltip: { show: false }, silent: true })
  }

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
    const labelPos = compact ? 'inside' : x < -20 ? 'left' : 'right'
    const ring = s ? (stale ? '#d4d4d8' : syncColor(s.sync_fraction)) : '#52525b'
    nodes.push({
      id: `w${rank}`,
      name: `rank ${rank}`,
      x,
      y,
      symbolSize: compact ? 44 : 52,
      itemStyle: {
        color: '#ffffff',
        borderColor: ring,
        borderWidth: compact ? 2.5 : 3,
        shadowColor: 'rgba(24, 24, 27, 0.08)',
        shadowBlur: 8,
      },
      label: {
        show: true,
        formatter: label(rank),
        position: labelPos,
        distance: 12,
        fontSize: compact ? 10 : 12,
        lineHeight: compact ? 12 : 18,
        color: '#18181b',
        fontWeight: compact ? 600 : 500,
        align: compact ? 'center' : labelPos === 'left' ? 'right' : 'left',
      },
    })
  }

  // A two-rank ring sits left/right to use the card's width; PS keeps workers
  // above/below so the push/pull edges stay long enough for their labels.
  const startAngle = workers === 2 && mode === 'ring' ? Math.PI : -Math.PI / 2

  if (mode === 'ps') {
    // ECharts trims edges by one radius per node (the mean of a [w, h] size),
    // so a wide capsule hides arrows on its sides. A disc clips evenly.
    nodes.push({
      id: 'ps',
      name: 'Parameter Server · CPU 聚合',
      x: 0,
      y: 0,
      symbol: 'circle',
      symbolSize: compact ? 42 : 78,
      itemStyle: {
        color: '#18181b',
        borderColor: '#ffffff',
        borderWidth: 2,
        shadowColor: 'rgba(24, 24, 27, 0.2)',
        shadowBlur: 10,
      },
      label: {
        show: true,
        formatter: compact ? 'PS' : '{t|PS}\n{s|CPU 聚合}',
        color: '#ffffff',
        fontWeight: 600,
        fontSize: 11,
        rich: {
          t: { color: '#ffffff', fontSize: 15, fontWeight: 600, lineHeight: 19 },
          s: { color: 'rgba(255, 255, 255, 0.7)', fontSize: 10, lineHeight: 14 },
        },
      },
    })
    for (let r = 0; r < workers; r += 1) {
      const angle = (2 * Math.PI * r) / workers + startAngle
      workerNode(r, Math.cos(angle) * radius, Math.sin(angle) * radius)
      links.push({
        source: `w${r}`,
        target: 'ps',
        lineStyle: { width: 1.6, curveness: 0.16, color: EDGE.push },
      })
      links.push({
        source: 'ps',
        target: `w${r}`,
        lineStyle: { width: 1.6, curveness: 0.16, type: 'dashed', color: EDGE.pull },
      })
    }
  } else if (mode === 'ring') {
    for (let r = 0; r < workers; r += 1) {
      const angle = (2 * Math.PI * r) / workers + startAngle
      workerNode(r, Math.cos(angle) * radius, Math.sin(angle) * radius)
    }
    for (let r = 0; r < workers && workers > 1; r += 1) {
      links.push({
        source: `w${r}`,
        target: `w${(r + 1) % workers}`,
        lineStyle: { width: 1.8, curveness: 0.2, color: EDGE.ring },
      })
    }
  } else {
    workerNode(0, 0, 0)
  }

  return {
    animation: false,
    tooltip: {
      formatter: (p: unknown) => {
        const item = p as { dataType?: string; data?: { name?: string } }
        return item.dataType === 'node' ? (item.data?.name ?? '') : ''
      },
    },
    series: [
      {
        type: 'graph',
        layout: 'none',
        roam: !compact,
        data: nodes,
        links,
        edgeSymbol: ['none', 'arrow'],
        edgeSymbolSize: 8,
        lineStyle: { color: '#a1a1aa', opacity: 0.9 },
        emphasis: { focus: 'adjacency' },
        top: compact ? 12 : 24,
        bottom: compact ? 12 : 24,
        left: 8,
        right: 8,
      },
    ],
  }
}

export default function TopologyTab({ detail, rankStatus, live }: { detail: RunDetail; rankStatus: RankStatus[]; live: boolean }) {
  const summary = detail.summary
  const option = useMemo(
    () =>
      topologyOption(
        summary.mode ?? 'single',
        summary.transport ?? 'none',
        summary.workers,
        rankStatus,
        false,
        summary.device,
      ),
    [summary.mode, summary.transport, summary.workers, rankStatus, summary.device],
  )

  if (detail.kind !== 'distributed') {
    return (
      <Panel>
        <EmptyState title="单进程训练没有通信拓扑" description="分布式运行（PS / Ring AllReduce）会在这里显示各 Rank 的连接与健康状态" />
      </Panel>
    )
  }

  const layout = `${summary.mode === 'ps' ? 'Parameter Server' : summary.mode === 'ring' ? 'Ring AllReduce' : '单进程'} · ${TRANSPORT_TEXT[summary.transport ?? ''] ?? summary.transport} · ${summary.workers} Worker${summary.device === 'cpu' ? '（CPU）' : '（共享单卡 GPU）'}`

  return (
    <Row gutter={[16, 16]}>
      <Col xs={24} xl={13}>
        <Panel
          title="通信拓扑"
          subtitle={layout}
          extra={
            <div className="inline-stats" style={{ fontSize: 'calc(12px * var(--s))', gap: '4px 14px' }}>
              {summary.mode === 'ps' ? (
                <>
                  <EdgeKey color={EDGE.push} text="Push 梯度" />
                  <EdgeKey color={EDGE.pull} dashed text="Pull 均值" />
                </>
              ) : summary.mode === 'ring' ? (
                <EdgeKey color={EDGE.ring} text="ScatterReduce + AllGather" />
              ) : null}
            </div>
          }
        >
          <div className="panel-body">
            <EChart option={option} height={380} />
          </div>
          <div className="panel-footer" style={{ justifyContent: 'flex-start', gap: 16, flexWrap: 'wrap' }}>
            <span>节点描边 = 同步占比</span>
            {SYNC_LEVELS.map(([color, text]) => (
              <span key={text} className="toolbar-item" style={{ gap: 6, fontSize: 'inherit', color: 'inherit' }}>
                <span className="status-dot" style={{ background: color }} />
                {text}
              </span>
            ))}
          </div>
        </Panel>
      </Col>

      <Col xs={24} xl={11}>
        <Panel title={live ? '各 Rank 实时状态' : '各 Rank 最终统计'}>
          <div style={{ padding: '8px 4px 4px' }}>
            <Table<RankStatus>
              size="small"
              rowKey="rank"
              pagination={false}
              dataSource={rankStatus}
              scroll={{ x: 'max-content' }}
              columns={[
                {
                  title: 'Rank',
                  dataIndex: 'rank',
                  render: (v) => (
                    <span className="toolbar-item" style={{ gap: 6, color: 'var(--text-main)', fontWeight: 600 }}>
                      <span className="stat-swatch" style={{ background: rankColor(v) }} />
                      rank {v}
                    </span>
                  ),
                },
                { title: 'PID', dataIndex: 'pid', render: (v) => <span className="mono muted">{v ?? '—'}</span> },
                { title: 'Epoch', dataIndex: 'epoch', render: (v) => (v === undefined ? '—' : `#${v + 1}`) },
                { title: '完成 Step', dataIndex: 'steps_done', align: 'right' },
                { title: 'Loss', dataIndex: 'loss', align: 'right', render: (v) => <span className="num">{fmtNum(v)}</span> },
                {
                  title: 'Step 耗时',
                  dataIndex: 'step_wall_s',
                  align: 'right',
                  render: (v) => <span className="num">{v ? `${(v * 1000).toFixed(1)} ms` : '—'}</span>,
                },
                {
                  title: '同步占比',
                  dataIndex: 'sync_fraction',
                  render: (v) => (v === null || v === undefined ? '—' : <SyncValue value={v} />),
                },
                ...(live
                  ? [
                      {
                        title: '心跳间隔',
                        dataIndex: 'seconds_since_last',
                        align: 'right' as const,
                        render: (v: number | null) => (
                          <span className="num">{v === null || v === undefined ? '—' : `${v.toFixed(1)} s`}</span>
                        ),
                      },
                    ]
                  : []),
              ]}
            />
          </div>
        </Panel>
      </Col>
    </Row>
  )
}
