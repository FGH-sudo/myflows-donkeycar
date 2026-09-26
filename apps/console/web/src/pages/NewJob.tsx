import { BranchesOutlined, CheckCircleFilled, ExclamationCircleFilled, RocketOutlined, ThunderboltOutlined } from '@ant-design/icons'
import type { ReactNode } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, App, Button, Col, Form, Input, InputNumber, Row, Segmented, Select, Space, Spin, Switch, Tabs, Tag } from 'antd'
import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import type { ModeOption, Preset } from '../api'
import EChart from '../components/EChart'
import EmptyState from '../components/EmptyState'
import { Panel } from '../components/Panel'
import { topologyOption } from './run/TopologyTab'

const FORM_KEYS = [
  'device',
  'optimizer',
  'learning_rate',
  'global_batch',
  'epochs',
  'steps',
  'seed',
  'timeout',
  'evaluate',
  'final_test',
  'profile',
  'monitor',
] as const
const HIDDEN_KEYS = new Set(['task', 'mode', 'transport', 'train_workers', ...FORM_KEYS])

interface TileOption {
  value: string
  title: ReactNode
  desc?: ReactNode
  badge?: ReactNode
}

/** Radio-style tiles; unlike Segmented they wrap instead of truncating labels. */
function OptionTiles({
  options,
  value,
  onChange,
  minWidth = 260,
}: {
  options: TileOption[]
  value?: string
  onChange: (v: string) => void
  minWidth?: number
}) {
  return (
    <div className="option-tiles" role="radiogroup" style={{ gridTemplateColumns: `repeat(auto-fit, minmax(min(${minWidth}px, 100%), 1fr))` }}>
      {options.map((o) => {
        const selected = o.value === value
        return (
          <button
            type="button"
            role="radio"
            aria-checked={selected}
            key={o.value}
            className={`option-tile${selected ? ' selected' : ''}`}
            onClick={() => onChange(o.value)}
          >
            <span className="option-tile-head">
              <span style={{ display: 'flex', alignItems: 'flex-start', gap: 10, minWidth: 0 }}>
                <span className="option-tile-check" />
                <span className="option-tile-title">{o.title}</span>
              </span>
              {o.badge}
            </span>
            {o.desc ? <span className="option-tile-desc" style={{ paddingLeft: 26 }}>{o.desc}</span> : null}
          </button>
        )
      })}
    </div>
  )
}

function PresetPicker({ presets, value, onChange }: { presets: Preset[]; value?: string; onChange: (id: string) => void }) {
  return (
    <OptionTiles
      value={value}
      onChange={onChange}
      minWidth={300}
      options={presets.map((p) => ({
        value: p.id,
        title: p.label,
        desc: p.description,
        badge: p.data_ready ? (
          <Tag color="success" icon={<CheckCircleFilled />} style={{ margin: 0 }}>
            数据就绪
          </Tag>
        ) : (
          <Tag color="warning" icon={<ExclamationCircleFilled />} style={{ margin: 0 }}>
            未就绪
          </Tag>
        ),
      }))}
    />
  )
}

const modeParts = (label: string) => {
  const [head, ...rest] = label.split(' · ')
  return { head, tail: rest.join(' · ') }
}


/** Numbered block inside the form panel; sections are separated by hairlines. */
function FormSection({ index, title, description, children }: { index: number; title: string; description?: ReactNode; children: ReactNode }) {
  return (
    <section className="form-section">
      <div className="form-section-head">
        <span className="form-section-index">{index}</span>
        <div>
          <div className="form-section-title">{title}</div>
          {description ? <div className="form-section-desc">{description}</div> : null}
        </div>
      </div>
      {children}
    </section>
  )
}

/** Right-hand summary rows. */
function SummaryList({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <dl className="summary-list">
      {rows.map(([k, v]) => (
        <div key={k}>
          <dt>{k}</dt>
          <dd>{v ?? '—'}</dd>
        </div>
      ))}
    </dl>
  )
}

function SubmitBox({
  placeholder,
  label,
  onLabel,
  disabled,
  loading,
  onSubmit,
  note,
}: {
  placeholder: string
  label: string
  onLabel: (v: string) => void
  disabled: boolean
  loading: boolean
  onSubmit: () => void
  note: ReactNode
}) {
  return (
    <div className="submit-box">
      <Input placeholder={placeholder} value={label} onChange={(e) => onLabel(e.target.value)} />
      <Button type="primary" block size="large" icon={<RocketOutlined />} loading={loading} disabled={disabled} onClick={onSubmit}>
        提交到训练队列
      </Button>
      <div className="submit-note">{note}</div>
    </div>
  )
}

function DistributedForm({
  presets,
  modes,
  workerChoices,
  maxWorkers,
}: {
  presets: Preset[]
  modes: ModeOption[]
  workerChoices: number[]
  maxWorkers: number
}) {
  const { message } = App.useApp()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [presetId, setPresetId] = useState(presets[0]?.id)
  const preset = presets.find((p) => p.id === presetId)
  const [modeKey, setModeKey] = useState('ps/grpc_proto')
  const [workers, setWorkers] = useState(2)
  const [label, setLabel] = useState('')
  const [advanced, setAdvanced] = useState('{}')
  const [submitting, setSubmitting] = useState(false)
  const [form] = Form.useForm()
  const [mode, transport] = modeKey.split('/')
  const modeLabel = modes.find((m) => `${m.mode}/${m.transport}` === modeKey)?.label ?? modeKey

  useEffect(() => {
    if (!preset?.config) return
    const cfg = preset.config
    form.setFieldsValue(Object.fromEntries(FORM_KEYS.map((k) => [k, cfg[k]])))
    setAdvanced(JSON.stringify(Object.fromEntries(Object.entries(cfg).filter(([k]) => !HIDDEN_KEYS.has(k))), null, 2))
  }, [preset, form])

  useEffect(() => {
    if (mode === 'single') setWorkers(1)
  }, [mode])

  const globalBatch = Form.useWatch('global_batch', form) as number | undefined
  const deviceValue = Form.useWatch('device', form) as string | undefined
  const optimizer = Form.useWatch('optimizer', form) as string | undefined
  const lr = Form.useWatch('learning_rate', form) as number | undefined
  const epochs = Form.useWatch('epochs', form) as number | undefined
  const device = deviceValue === 'cpu' ? 'CPU' : 'GPU'
  const batchError =
    globalBatch !== undefined && globalBatch < workers ? `全局 batch（${globalBatch}）必须不小于 Worker 数（${workers}）` : null
  const preview = useMemo(() => topologyOption(mode, transport, workers, [], true), [mode, transport, workers])

  const submit = async () => {
    if (!preset) return
    let extra: Record<string, unknown>
    try {
      extra = JSON.parse(advanced || '{}')
    } catch {
      message.error('高级参数不是合法 JSON')
      return
    }
    const values = await form.validateFields()
    const config = { ...extra, ...values, task: preset.task, mode, transport, train_workers: workers }
    setSubmitting(true)
    try {
      const job = await api.submitJob({ type: 'distributed', config, label: label || undefined })
      message.success(`已提交任务：${job.label}`)
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
      queryClient.invalidateQueries({ queryKey: ['runs'] })
      navigate(`/run/console/${job.id}`)
    } catch (e) {
      message.error((e as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="checkout">
      <Panel>
        <FormSection index={1} title="任务预设" description="预设决定数据集与默认超参数，可在下方继续调整">
          <PresetPicker presets={presets} value={presetId} onChange={setPresetId} />
          {preset && !preset.data_ready ? <Alert style={{ marginTop: 12 }} type="warning" showIcon message={preset.data_hint} /> : null}
        </FormSection>

        <FormSection index={2} title="分布式架构" description="仅列出框架已验证的模式与通信组合">
          <OptionTiles
            value={modeKey}
            onChange={setModeKey}
            minWidth={132}
            options={modes.map((m) => {
              const { head, tail } = modeParts(m.label)
              return { value: `${m.mode}/${m.transport}`, title: head, desc: tail || '无通信' }
            })}
          />
          <div style={{ marginTop: 20 }}>
            <span className="section-label">{device === 'GPU' ? 'Worker 数（所有 Worker 共享本机单张 GPU）' : 'Worker 进程数'}</span>
            <Space wrap size={8}>
              <Segmented
                value={workerChoices.includes(workers) ? workers : 'custom'}
                disabled={mode === 'single'}
                onChange={(v) => v !== 'custom' && setWorkers(Number(v))}
                options={[...workerChoices.map((w) => ({ value: w, label: `${w} 个` })), { value: 'custom', label: '自定义' }]}
              />
              <InputNumber min={1} max={maxWorkers} value={workers} disabled={mode === 'single'} onChange={(v) => v && setWorkers(v)} style={{ width: 96 }} />
            </Space>
          </div>
          {batchError ? <Alert style={{ marginTop: 16 }} type="error" showIcon message={batchError} /> : null}
        </FormSection>

        <FormSection index={3} title="超参数">
          <Form form={form} name="distributed" layout="vertical" requiredMark={false}>
            <Row gutter={16}>
              <Col xs={24} sm={12} md={8}>
                <Form.Item name="device" label="计算设备">
                  <Select options={[{ value: 'cuda', label: 'CUDA (GPU)' }, { value: 'cpu', label: 'CPU' }]} />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12} md={8}>
                <Form.Item name="optimizer" label="优化器">
                  <Select options={[{ value: 'adam', label: 'Adam' }, { value: 'mbgd', label: 'MBGD' }]} />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12} md={8}>
                <Form.Item name="learning_rate" label="学习率" rules={[{ required: true }]}>
                  <InputNumber min={1e-7} step={1e-4} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12} md={8}>
                <Form.Item name="global_batch" label="全局 Batch" rules={[{ required: true }]}>
                  <InputNumber min={1} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12} md={8}>
                <Form.Item name="epochs" label="Epoch 轮数" rules={[{ required: true }]}>
                  <InputNumber min={1} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12} md={8}>
                <Form.Item name="steps" label={preset?.task === 'synthetic' ? '总 Step 数' : '每 Epoch 最大 Batch'} rules={[{ required: true }]}>
                  <InputNumber min={1} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12} md={8}>
                <Form.Item name="seed" label="随机种子">
                  <InputNumber style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12} md={8}>
                <Form.Item name="timeout" label="超时时间（秒）">
                  <InputNumber min={1} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
            </Row>
            <div className="switch-panel">
              <Form.Item name="evaluate" label="验证集评估" valuePropName="checked">
                <Switch />
              </Form.Item>
              <Form.Item name="final_test" label="测试集评估" valuePropName="checked">
                <Switch />
              </Form.Item>
              <Form.Item name="profile" label="CUDA Event 计时" valuePropName="checked">
                <Switch />
              </Form.Item>
              <Form.Item name="monitor" label="硬件资源采样" valuePropName="checked">
                <Switch />
              </Form.Item>
            </div>
          </Form>
        </FormSection>

        <FormSection index={4} title="高级参数" description="JSON，会与上方表单合并；表单字段优先">
          <Input.TextArea
            value={advanced}
            onChange={(e) => setAdvanced(e.target.value)}
            autoSize={{ minRows: 4, maxRows: 16 }}
            style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}
          />
        </FormSection>
      </Panel>

      <aside className="checkout-side">
        <Panel title="拓扑预览" subtitle={modeLabel}>
          <div className="panel-body">
            <EChart option={preview} height={220} />
          </div>
          <div className="panel-footer" style={{ lineHeight: 1.6 }}>
            {mode === 'ps'
              ? `1 个 CPU Parameter Server 聚合梯度，${workers} 个 ${device} Worker 本地更新参数。`
              : mode === 'ring'
                ? `${workers} 个对等 Rank，ScatterReduce + AllGather 环形规约，无单点瓶颈。`
                : '单进程运行基线，用于与分布式方案对照。'}
          </div>
        </Panel>

        <Panel title="配置摘要">
          <SummaryList
            rows={[
              ['任务', preset?.label],
              ['架构', modeLabel],
              ['Worker', `${workers} 个 · ${device}`],
              ['优化器 / 学习率', optimizer ? <span className="num" key="o">{`${optimizer} / ${lr ?? '—'}`}</span> : null],
              ['全局 Batch / Epoch', <span className="num" key="b">{`${globalBatch ?? '—'} / ${epochs ?? '—'}`}</span>],
            ]}
          />
          <SubmitBox
            placeholder="任务名称（可选），如 resnet18_ring_2w_lr1e-3"
            label={label}
            onLabel={setLabel}
            disabled={!preset || !preset.data_ready || !!batchError}
            loading={submitting}
            onSubmit={submit}
            note={
              <>
                按提交顺序在单卡上串行执行，产物写入 <code>runs/console/</code>
              </>
            }
          />
        </Panel>
      </aside>
    </div>
  )
}

function SingleForm({ presets }: { presets: Preset[] }) {
  const { message } = App.useApp()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [presetId, setPresetId] = useState(presets[0]?.id)
  const preset = presets.find((p) => p.id === presetId)
  const [form] = Form.useForm()
  const [label, setLabel] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const epochs = Form.useWatch('epochs', form) as number | undefined
  const batch = Form.useWatch('batch', form) as number | undefined
  const lr = Form.useWatch('lr', form) as number | undefined
  const device = Form.useWatch('device', form) as string | undefined
  const maxSamples = Form.useWatch('max_samples', form) as number | undefined

  useEffect(() => {
    if (preset?.args) form.setFieldsValue({ augment: false, graph_opt: false, ...preset.args })
  }, [preset, form])

  const submit = async () => {
    const values = await form.validateFields()
    const args = Object.fromEntries(Object.entries(values).filter(([, v]) => v !== undefined && v !== null && v !== ''))
    setSubmitting(true)
    try {
      const job = await api.submitJob({ type: 'single', args, label: label || undefined })
      message.success(`已提交任务：${job.label}`)
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
      navigate(`/run/console/${job.id}`)
    } catch (e) {
      message.error((e as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="checkout">
      <Panel>
        <FormSection index={1} title="任务预设" description={<code>apps.train.train_myflows_donkey</code>}>
          <PresetPicker presets={presets} value={presetId} onChange={setPresetId} />
        </FormSection>

        <FormSection index={2} title="超参数与数据">
          <Form form={form} name="single" layout="vertical" requiredMark={false}>
            <Row gutter={16}>
              <Col xs={24} sm={12} md={8}>
                <Form.Item name="max_samples" label="样本数（0 = 全量）">
                  <InputNumber min={0} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12} md={8}>
                <Form.Item name="epochs" label="Epoch 轮数" rules={[{ required: true }]}>
                  <InputNumber min={1} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12} md={8}>
                <Form.Item name="batch" label="Batch 大小" rules={[{ required: true }]}>
                  <InputNumber min={1} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12} md={8}>
                <Form.Item name="lr" label="学习率">
                  <InputNumber min={1e-7} step={1e-4} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12} md={8}>
                <Form.Item name="device" label="设备">
                  <Select options={['auto', 'cuda', 'cpu'].map((v) => ({ value: v, label: v }))} />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12} md={8}>
                <Form.Item name="dtype" label="精度">
                  <Select options={['float32', 'float64'].map((v) => ({ value: v, label: v }))} />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12} md={8}>
                <Form.Item name="val_size" label="验证集样本数">
                  <InputNumber min={0} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12} md={8}>
                <Form.Item name="checkpoint_every" label="断点保存频率（step）">
                  <InputNumber min={0} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12} md={8}>
                <Form.Item name="tb_log_interval" label="TensorBoard 记录间隔">
                  <InputNumber min={1} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
            </Row>
            <div className="switch-panel">
              <Form.Item name="augment" label="数据增强" valuePropName="checked">
                <Switch />
              </Form.Item>
              <Form.Item name="graph_opt" label="计算图优化" valuePropName="checked">
                <Switch />
              </Form.Item>
            </div>
          </Form>
        </FormSection>
      </Panel>

      <aside className="checkout-side">
        <Panel title="配置摘要" subtitle="单进程 ResNet18 基线">
          <SummaryList
            rows={[
              ['任务', preset?.label],
              ['样本数', maxSamples ? <span className="num" key="m">{maxSamples}</span> : '全量'],
              ['Epoch / Batch', <span className="num" key="e">{`${epochs ?? '—'} / ${batch ?? '—'}`}</span>],
              ['学习率', lr !== undefined ? <span className="num" key="l">{lr}</span> : null],
              ['设备', device],
            ]}
          />
          <SubmitBox
            placeholder="任务名称（可选），如 resnet18_baseline_e5"
            label={label}
            onLabel={setLabel}
            disabled={!preset?.data_ready}
            loading={submitting}
            onSubmit={submit}
            note="每步写入 metrics.jsonl，控制台每 500 ms 采样整卡硬件；停止任务时先软退出并保存最新断点。"
          />
        </Panel>
      </aside>
    </div>
  )
}

export default function NewJob() {
  const presets = useQuery({ queryKey: ['presets'], queryFn: api.presets })
  if (presets.isLoading) return <Spin style={{ display: 'block', marginTop: 80 }} />
  if (!presets.data) {
    return (
      <Panel>
        <EmptyState title="无法加载任务预设" description="请确认控制台后端正在运行" />
      </Panel>
    )
  }
  const p = presets.data

  return (
    <div>
      <div className="page-title">
        <div>
          <h2>新建训练</h2>
          <div className="page-subtitle">配置分布式（Parameter Server / Ring AllReduce）或单进程 ResNet 训练</div>
        </div>
      </div>
      <Tabs
        items={[
          {
            key: 'dist',
            label: (
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                <BranchesOutlined />
                分布式训练
              </span>
            ),
            children: <DistributedForm presets={p.distributed} modes={p.modes} workerChoices={p.worker_choices} maxWorkers={p.max_workers} />,
          },
          {
            key: 'single',
            label: (
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                <ThunderboltOutlined />
                单进程基线
              </span>
            ),
            children: <SingleForm presets={p.single} />,
          },
        ]}
      />
    </div>
  )
}
