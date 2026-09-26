import { CopyOutlined } from '@ant-design/icons'
import { App, Button, Col, Row, Table } from 'antd'
import type { ReactNode } from 'react'
import type { RunDetail, SplitMetrics } from '../../api'
import EmptyState from '../../components/EmptyState'
import { Panel } from '../../components/Panel'
import StatusTag from '../../components/StatusTag'
import { KIND_TEXT, fmtDuration, fmtNum, fmtTime, modeLabel } from '../../format'

const METRIC_KEYS: [keyof SplitMetrics, string][] = [
  ['samples', '样本数'],
  ['loss', 'loss'],
  ['accuracy', 'accuracy'],
  ['mse', 'MSE'],
  ['angle_mae', 'angle MAE'],
  ['angle_rmse', 'angle RMSE'],
  ['throttle_mae', 'throttle MAE'],
]

const EPOCH_COLUMNS: [string, string, (v: number) => string][] = [
  ['epoch_train_wall_s', 'epoch 时间', (v) => fmtDuration(v)],
  ['samples_per_s', 'samples/s', (v) => fmtNum(v, 0)],
  ['gradient_sync_s', '同步耗时', (v) => fmtDuration(v)],
  ['forward_gpu_s', 'forward GPU', (v) => fmtDuration(v)],
  ['backward_gpu_s', 'backward GPU', (v) => fmtDuration(v)],
  ['optimizer_wall_s', 'optimizer', (v) => fmtDuration(v)],
  ['val_loss', 'val loss', (v) => fmtNum(v)],
  ['val_accuracy', 'val acc', (v) => fmtNum(v)],
  ['val_angle_mae', 'val angle MAE', (v) => fmtNum(v)],
  ['loss', 'train loss', (v) => fmtNum(v)],
  ['epoch_time_s', 'epoch 时间', (v) => fmtDuration(v)],
]

/** Borderless label/value grid; `wide` items span the full row. */
function KvGrid({ items }: { items: { label: string; value: ReactNode; wide?: boolean }[] }) {
  return (
    <dl className="kv-grid">
      {items.map((it) => (
        <div key={it.label} className={`kv-item${it.wide ? ' wide' : ''}`}>
          <dt>{it.label}</dt>
          <dd>{it.value}</dd>
        </div>
      ))}
    </dl>
  )
}

export default function SummaryTab({ detail }: { detail: RunDetail }) {
  const { message } = App.useApp()
  const s = detail.summary
  const final = (detail.results?.final ?? {}) as Record<string, SplitMetrics | unknown>
  const splits = (['train', 'val', 'test'] as const).filter((k) => final[k] && typeof final[k] === 'object')
  const epochRows = detail.epochs ?? []
  const epochCols = EPOCH_COLUMNS.filter(([k]) => epochRows.some((r) => typeof r[k] === 'number'))
  const seen = new Set<string>()
  const uniqueEpochCols = epochCols.filter(([, label]) => (seen.has(label) ? false : (seen.add(label), true)))
  const manifest = detail.manifest as Record<string, unknown> | null
  const packages = (manifest?.packages ?? {}) as Record<string, string>

  const copyConfig = () => {
    if (detail.config) {
      navigator.clipboard.writeText(JSON.stringify(detail.config, null, 2))
      message.success('配置 JSON 已复制')
    }
  }

  return (
    <Row gutter={[16, 16]}>
      <Col xs={24} xl={14}>
        <Panel title="运行信息与超参数" style={{ height: '100%' }}>
          <KvGrid
            items={[
              { label: '状态', value: <StatusTag status={s.status} /> },
              { label: '类型', value: KIND_TEXT[s.kind] ?? s.kind },
              { label: '任务', value: s.task ?? '—' },
              { label: '架构与协议', value: modeLabel(s) },
              { label: 'Worker 数', value: s.workers },
              { label: '计算设备', value: s.device ?? '—' },
              { label: '优化器 / 学习率', value: <span className="num">{`${s.optimizer ?? '—'} / ${s.learning_rate ?? '—'}`}</span> },
              { label: '全局 Batch / Epoch', value: <span className="num">{`${s.global_batch ?? '—'} / ${s.epochs ?? '—'}`}</span> },
              { label: '启动时间', value: <span className="num">{fmtTime(s.started_unix_s)}</span> },
              { label: '运行总耗时', value: <span className="num">{fmtDuration(s.elapsed_s)}</span> },
              { label: '产物路径', value: <code style={{ wordBreak: 'break-all' }}>{s.path}</code>, wide: true },
              ...(s.error ? [{ label: '失败原因', value: <span style={{ color: 'var(--red)' }}>{s.error}</span>, wide: true }] : []),
            ]}
          />
        </Panel>
      </Col>

      <Col xs={24} xl={10}>
        <Panel title="最终评估指标" style={{ height: '100%' }}>
          {splits.length ? (
            <div style={{ padding: '8px 4px 4px' }}>
              <Table
                size="small"
                pagination={false}
                rowKey="key"
                dataSource={METRIC_KEYS.filter(([k]) => splits.some((sp) => (final[sp] as SplitMetrics)[k] !== undefined)).map(
                  ([k, label]) => ({
                    key: k,
                    label,
                    ...Object.fromEntries(splits.map((sp) => [sp, (final[sp] as SplitMetrics)[k]])),
                  }),
                )}
                columns={[
                  { title: '指标', dataIndex: 'label', render: (v) => <span style={{ color: 'var(--text-secondary)' }}>{v}</span> },
                  ...splits.map((sp) => ({
                    title: sp,
                    dataIndex: sp,
                    align: 'right' as const,
                    render: (v: number) => (v !== undefined ? <span className="num" style={{ fontWeight: 600 }}>{fmtNum(v)}</span> : '—'),
                  })),
                ]}
              />
            </div>
          ) : s.final && Object.keys(s.final).length ? (
            <KvGrid
              items={Object.entries(s.final).map(([k, v]) => ({
                label: k,
                value: <span className="num" style={{ fontWeight: 600 }}>{typeof v === 'number' ? fmtNum(v) : String(v)}</span>,
              }))}
            />
          ) : (
            <EmptyState compact title="训练结束后显示评估指标" />
          )}
        </Panel>
      </Col>

      {epochRows.length ? (
        <Col span={24}>
          <Panel title="逐 Epoch 指标">
            <div style={{ padding: '8px 4px 4px' }}>
              <Table
                size="small"
                pagination={false}
                rowKey={(r) => String(r.epoch)}
                dataSource={epochRows}
                scroll={{ x: 'max-content' }}
                columns={[
                  {
                    title: 'Epoch',
                    dataIndex: 'epoch',
                    width: 80,
                    render: (v: number) => <span style={{ fontWeight: 600 }}>#{detail.kind === 'distributed' ? v + 1 : v}</span>,
                  },
                  ...uniqueEpochCols.map(([k, label, fmt]) => ({
                    title: label,
                    dataIndex: k,
                    align: 'right' as const,
                    render: (v: number) => (typeof v === 'number' ? <span className="num">{fmt(v)}</span> : '—'),
                  })),
                ]}
              />
            </div>
          </Panel>
        </Col>
      ) : null}

      <Col xs={24} xl={12}>
        <Panel
          title="训练配置"
          subtitle="config.json"
          extra={
            <Button size="small" type="text" icon={<CopyOutlined />} onClick={copyConfig} disabled={!detail.config}>
              复制
            </Button>
          }
          padded
        >
          <pre className="json-view">{JSON.stringify(detail.config ?? {}, null, 2)}</pre>
        </Panel>
      </Col>

      <Col xs={24} xl={12}>
        <Panel title="环境与版本快照" subtitle={manifest ? 'manifest.json' : undefined} style={{ height: '100%' }}>
          {manifest ? (
            <KvGrid
              items={[
                { label: '运行平台', value: String(manifest.platform ?? '—'), wide: true },
                { label: 'Python', value: <span className="num">{String(manifest.python ?? '—').split(' ')[0]}</span> },
                { label: '根仓库 Commit', value: <code>{String(manifest.root_head ?? '—').slice(0, 12)}</code> },
                { label: 'MyFlows Commit', value: <code>{String(manifest.myflows_head ?? '—').slice(0, 12)}</code> },
                ...Object.entries(packages).map(([k, v]) => ({ label: k, value: <span className="mono">{v}</span> })),
              ]}
            />
          ) : detail.job ? (
            <div style={{ padding: '12px 20px 20px' }}>
              <span className="section-label">启动命令</span>
              <pre className="json-view">{detail.job.command.join(' ')}</pre>
            </div>
          ) : (
            <EmptyState compact title="没有关联的 manifest 快照" />
          )}
        </Panel>
      </Col>
    </Row>
  )
}
