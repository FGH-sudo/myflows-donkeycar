import { Card, Col, Descriptions, Empty, Row, Table } from 'antd'
import type { RunDetail, SplitMetrics } from '../../api'
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
  ['gradient_sync_s', '同步', (v) => fmtDuration(v)],
  ['forward_gpu_s', 'forward GPU', (v) => fmtDuration(v)],
  ['backward_gpu_s', 'backward GPU', (v) => fmtDuration(v)],
  ['optimizer_wall_s', 'optimizer', (v) => fmtDuration(v)],
  ['val_loss', 'val loss', (v) => fmtNum(v)],
  ['val_accuracy', 'val acc', (v) => fmtNum(v)],
  ['val_angle_mae', 'val angle MAE', (v) => fmtNum(v)],
  ['loss', 'train loss', (v) => fmtNum(v)],
  ['epoch_time_s', 'epoch 时间', (v) => fmtDuration(v)],
]

export default function SummaryTab({ detail }: { detail: RunDetail }) {
  const s = detail.summary
  const final = (detail.results?.final ?? {}) as Record<string, SplitMetrics | unknown>
  const splits = (['train', 'val', 'test'] as const).filter((k) => final[k] && typeof final[k] === 'object')
  const epochRows = detail.epochs ?? []
  const epochCols = EPOCH_COLUMNS.filter(([k]) => epochRows.some((r) => typeof r[k] === 'number'))
  const seen = new Set<string>()
  const uniqueEpochCols = epochCols.filter(([, label]) => (seen.has(label) ? false : (seen.add(label), true)))
  const manifest = detail.manifest as Record<string, unknown> | null
  const packages = (manifest?.packages ?? {}) as Record<string, string>

  return (
    <Row gutter={[16, 16]}>
      <Col xs={24} xl={14}>
        <Card size="small" title="运行信息">
          <Descriptions size="small" column={2} bordered items={[
            { key: 'status', label: '状态', children: <StatusTag status={s.status} /> },
            { key: 'kind', label: '类型', children: KIND_TEXT[s.kind] ?? s.kind },
            { key: 'task', label: '任务', children: s.task ?? '—' },
            { key: 'mode', label: '模式', children: modeLabel(s) },
            { key: 'workers', label: 'Worker 数', children: s.workers },
            { key: 'device', label: '设备', children: s.device ?? '—' },
            { key: 'optimizer', label: '优化器 / 学习率', children: `${s.optimizer ?? '—'} / ${s.learning_rate ?? '—'}` },
            { key: 'batch', label: '全局 batch / epoch', children: `${s.global_batch ?? '—'} / ${s.epochs ?? '—'}` },
            { key: 'start', label: '开始', children: fmtTime(s.started_unix_s) },
            { key: 'elapsed', label: '耗时', children: fmtDuration(s.elapsed_s) },
            { key: 'path', label: '目录', span: 2, children: <code style={{ fontSize: 12 }}>{s.path}</code> },
            ...(s.error ? [{ key: 'error', label: '错误', span: 2, children: <span style={{ color: '#cf1322' }}>{s.error}</span> }] : []),
          ]} />
        </Card>
      </Col>
      <Col xs={24} xl={10}>
        <Card size="small" title="最终指标">
          {splits.length ? (
            <Table size="small" pagination={false} rowKey="key"
              dataSource={METRIC_KEYS.filter(([k]) => splits.some((sp) => (final[sp] as SplitMetrics)[k] !== undefined))
                .map(([k, label]) => ({ key: k, label, ...Object.fromEntries(splits.map((sp) => [sp, (final[sp] as SplitMetrics)[k]])) }))}
              columns={[{ title: '指标', dataIndex: 'label' }, ...splits.map((sp) => ({ title: sp, dataIndex: sp, render: (v: number) => (sp && v !== undefined ? fmtNum(v) : '—') }))]} />
          ) : s.final && Object.keys(s.final).length ? (
            <Descriptions size="small" column={1} items={Object.entries(s.final).map(([k, v]) => ({ key: k, label: k, children: typeof v === 'number' ? fmtNum(v) : String(v) }))} />
          ) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="运行结束后显示" />}
        </Card>
      </Col>
      {epochRows.length ? (
        <Col span={24}>
          <Card size="small" title="逐 epoch 汇总">
            <Table size="small" pagination={false} rowKey={(r) => String(r.epoch)} dataSource={epochRows} scroll={{ x: true }}
              columns={[{ title: 'epoch', dataIndex: 'epoch', width: 70, render: (v: number) => (detail.kind === 'distributed' ? v + 1 : v) },
                ...uniqueEpochCols.map(([k, label, fmt]) => ({ title: label, dataIndex: k, render: (v: number) => (typeof v === 'number' ? fmt(v) : '—') }))]} />
          </Card>
        </Col>
      ) : null}
      <Col xs={24} xl={12}>
        <Card size="small" title="配置">
          <pre className="json-view">{JSON.stringify(detail.config ?? {}, null, 2)}</pre>
        </Card>
      </Col>
      <Col xs={24} xl={12}>
        <Card size="small" title="环境与版本">
          {manifest ? (
            <Descriptions size="small" column={1} items={[
              { key: 'platform', label: '平台', children: String(manifest.platform ?? '—') },
              { key: 'python', label: 'Python', children: String(manifest.python ?? '—').split(' ')[0] },
              { key: 'root', label: '根仓库 HEAD', children: <code>{String(manifest.root_head ?? '—').slice(0, 12)}</code> },
              { key: 'myflows', label: 'MyFlows HEAD', children: <code>{String(manifest.myflows_head ?? '—').slice(0, 12)}</code> },
              ...Object.entries(packages).map(([k, v]) => ({ key: k, label: k, children: v })),
            ]} />
          ) : detail.job ? (
            <pre className="json-view">{detail.job.command.join(' ')}</pre>
          ) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="没有 manifest" />}
        </Card>
      </Col>
    </Row>
  )
}
