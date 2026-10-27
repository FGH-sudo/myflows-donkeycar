import { BranchesOutlined, CheckCircleFilled, ExclamationCircleFilled, RocketOutlined, ThunderboltOutlined } from '@ant-design/icons'
import type { ReactNode } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, App, Button, Col, Form, Input, InputNumber, Row, Segmented, Select, Space, Spin, Switch, Tabs, Tag } from 'antd'
import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import type { ConvBackendOption, ModeOption, NativeLibraryStatus, Preset } from '../api'
import EChart from '../components/EChart'
import EmptyState from '../components/EmptyState'
import { Panel } from '../components/Panel'
import { topologyOption } from './run/TopologyTab'

const FORM_KEYS = [
  'device',
  'conv_backend',
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

const CONV_TOOLTIP = '卷积与池化的实现。两个原生选项只有 GEMM 引擎不同：寄存器分块是第 19 讲的自写 kernel，cuBLAS 是 NVIDIA 库'

interface ConvChoice {
  value: string
  label: string
  description: string
}

/** Conv backend choices for a device; `auto` (apps.train only) resolves to register-tiled on CUDA + float32. */
function convOptions(backends: ConvBackendOption[], device: string | undefined, withAuto: boolean, float64 = false): ConvChoice[] {
  const options = backends
    .filter((b) => device === undefined || device === 'auto' || b.device === device)
    .filter((b) => !(float64 && b.value.startsWith('cuda_native')))
    .map((b) => ({ value: b.value, label: b.label, description: b.default ? `默认 · ${b.description}` : b.description }))
  const auto = { value: 'auto', label: '自动', description: 'CUDA + float32 用原生 · 寄存器分块（缺失时先自动编译），float64 用 CuPy，CPU 用 NumPy' }
  return withAuto ? [auto, ...options] : options
}

/** Select whose dropdown adds each option's description under its label. */
function ConvSelect({ options, ...rest }: { options: ConvChoice[]; value?: string; onChange?: (v: string) => void }) {
  return (
    <Select
      {...rest}
      options={options}
      popupMatchSelectWidth={false}
      optionRender={(o) => (
        <div>
          <div>{o.label}</div>
          <div className="select-option-desc">{o.data.description}</div>
        </div>
      )}
    />
  )
}

const NATIVE_REASONS: Record<string, string> = {
  missing: '尚未编译',
  'sources changed': '与当前源码不一致',
  'sources newer than library': '比源码旧',
}

/** Shown when a native backend is selected but the library is not built from the current sources. */
function NativeLibraryNotice({ status }: { status: NativeLibraryStatus }) {
  if (status.current) return null
  const reason = (status.reason && NATIVE_REASONS[status.reason]) ?? status.reason ?? '不可用'
  return status.cmake ? (
    <Alert
      style={{ marginBottom: 16 }}
      type="info"
      showIcon
      message={`原生 CUDA 库${reason}，任务启动后会先自动编译（需要 C++ 编译器，首次约 1–2 分钟），编译输出见任务日志。`}
    />
  ) : (
    <Alert
      style={{ marginBottom: 16 }}
      type="warning"
      showIcon
      message={`原生 CUDA 库${reason}，且本机未找到 CMake，自动编译会失败。请先安装 CMake 与 C++ 编译器（见 docs/development_environment.md），或改选 CuPy。`}
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
  convBackends,
  nativeLibrary,
}: {
  presets: Preset[]
  modes: ModeOption[]
  workerChoices: number[]
  maxWorkers: number
  convBackends: ConvBackendOption[]
  nativeLibrary: NativeLibraryStatus
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
  const convBackend = Form.useWatch('conv_backend', form) as string | undefined
  const device = deviceValue === 'cpu' ? 'CPU' : 'GPU'
  // Only the donkey_* tasks have convolutions; their backend follows the device.
  const convTask = preset?.task.startsWith('donkey_') ?? false
  const convChoices = useMemo(() => convOptions(convBackends, deviceValue ?? 'cuda', false), [convBackends, deviceValue])

  useEffect(() => {
    if (convTask && deviceValue && !convChoices.some((o) => o.value === convBackend)) {
      const fallback = convBackends.find((b) => b.device === deviceValue && b.default)
      form.setFieldValue('conv_backend', fallback?.value)
    }
  }, [convTask, deviceValue, convBackend, convChoices, convBackends, form])
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
        <FormSection
          index={1}
          title="任务预设"
          description="分布式实验用冻结的数据与配置逐步记录计时和通信量，用来对比 PS / Ring；要训练可部署的模型请用“正式训练”"
        >
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
              {convTask ? (
                <Col xs={24} sm={12} md={8}>
                  <Form.Item name="conv_backend" label="卷积后端" tooltip={CONV_TOOLTIP}>
                    <ConvSelect options={convChoices} />
                  </Form.Item>
                </Col>
              ) : null}
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
            {convTask && convBackend?.startsWith('cuda_native') ? <NativeLibraryNotice status={nativeLibrary} /> : null}
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
              ...(convTask ? [['卷积后端', convBackends.find((b) => b.value === convBackend)?.label] as [string, ReactNode]] : []),
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

/** Parallel modes of apps.train: same training loop, checkpoints and ONNX; `distributed/transport` args. */
const TRAIN_PARALLEL: TileOption[] = [
  { value: 'none/none', title: '单进程', desc: '一个进程完成训练' },
  { value: 'ring/grpc_proto', title: 'Ring AllReduce', desc: 'gRPC · 环形规约梯度' },
  { value: 'ps/grpc_proto', title: 'Parameter Server', desc: 'gRPC · 参数服务器汇总梯度' },
  { value: 'ps/socket_json', title: 'Parameter Server', desc: 'Socket JSON · ResNet 宽度 ≤ 32' },
]

function TrainingForm({
  presets,
  convBackends,
  nativeLibrary,
  workerChoices,
  maxWorkers,
}: {
  presets: Preset[]
  convBackends: ConvBackendOption[]
  nativeLibrary: NativeLibraryStatus
  workerChoices: number[]
  maxWorkers: number
}) {
  const { message } = App.useApp()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [presetId, setPresetId] = useState(presets[0]?.id)
  const preset = presets.find((p) => p.id === presetId)
  const [form] = Form.useForm()
  const [label, setLabel] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [parallelKey, setParallelKey] = useState('none/none')
  const [workers, setWorkers] = useState(2)
  const [distributed, transport] = parallelKey.split('/')
  const parallel = distributed !== 'none'
  const epochs = Form.useWatch('epochs', form) as number | undefined
  const batch = Form.useWatch('batch', form) as number | undefined
  const lr = Form.useWatch('lr', form) as number | undefined
  const device = Form.useWatch('device', form) as string | undefined
  const maxSamples = Form.useWatch('max_samples', form) as number | undefined
  const dtype = Form.useWatch('dtype', form) as string | undefined
  const convBackend = Form.useWatch('conv_backend', form) as string | undefined
  const baseWidth = Form.useWatch('base_width', form) as number | undefined
  const convChoices = useMemo(
    () => convOptions(convBackends, device, true, dtype === 'float64'),
    [convBackends, device, dtype],
  )
  // auto picks the native register-tiled backend on CUDA + float32 (device auto may resolve to CUDA).
  const usesNative =
    convBackend?.startsWith('cuda_native') || (convBackend === 'auto' && device !== 'cpu' && dtype !== 'float64')
  const preview = useMemo(
    () => topologyOption(parallel ? distributed : 'single', transport, parallel ? workers : 1, [], true),
    [parallel, distributed, transport, workers],
  )
  const shards = useMemo(() => {
    if (!parallel || !batch || batch < workers) return null
    const base = Math.floor(batch / workers)
    return Array.from({ length: workers }, (_, i) => base + (i < batch % workers ? 1 : 0))
  }, [parallel, batch, workers])
  const parallelError = !parallel
    ? null
    : batch !== undefined && batch < workers
      ? `全局 Batch（${batch}）必须不小于 Worker 数（${workers}）`
      : dtype === 'float64'
        ? '分布式训练按 FP32 交换梯度，只支持 float32'
        : transport === 'socket_json' && (baseWidth ?? 64) > 32
          ? 'Socket JSON 单条消息有 64 MB 上限，只能训练 ResNet 宽度不超过 32 的模型；请改用 gRPC'
          : null

  useEffect(() => {
    if (!preset?.args) return
    const { distributed: d = 'none', transport: t, workers: w, ...rest } = preset.args as Record<string, unknown>
    form.setFieldsValue({ augment: false, graph_opt: false, export_onnx: false, early_stopping: false, conv_backend: 'auto', ...rest })
    setParallelKey(d === 'none' ? 'none/none' : `${d}/${t ?? 'grpc_proto'}`)
    if (typeof w === 'number' && w > 0) setWorkers(w)
  }, [preset, form])

  useEffect(() => {
    if (convBackend && !convChoices.some((o) => o.value === convBackend)) form.setFieldValue('conv_backend', 'auto')
  }, [convBackend, convChoices, form])

  const submit = async () => {
    const values = await form.validateFields()
    const args: Record<string, unknown> = Object.fromEntries(
      Object.entries(values).filter(([, v]) => v !== undefined && v !== null && v !== ''),
    )
    args.distributed = distributed
    if (parallel) Object.assign(args, { transport, workers })
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

        <FormSection
          index={2}
          title="并行方式"
          description="多 Worker 时 Batch 为全局大小、按 Worker 切分；每步按样本数平均梯度与 BN 统计量，断点、最佳模型与 ONNX 和单进程相同"
        >
          <OptionTiles value={parallelKey} onChange={setParallelKey} minWidth={150} options={TRAIN_PARALLEL} />
          {parallel ? (
            <div style={{ marginTop: 20 }}>
              <span className="section-label">Worker 数（所有 Worker 共享本机 GPU）</span>
              <Space wrap size={8}>
                <Segmented
                  value={workerChoices.includes(workers) ? workers : 'custom'}
                  onChange={(v) => v !== 'custom' && setWorkers(Number(v))}
                  options={[...workerChoices.map((w) => ({ value: w, label: `${w} 个` })), { value: 'custom', label: '自定义' }]}
                />
                <InputNumber min={1} max={maxWorkers} value={workers} onChange={(v) => v && setWorkers(v)} style={{ width: 96 }} />
              </Space>
            </div>
          ) : null}
          {parallelError ? <Alert style={{ marginTop: 16 }} type="error" showIcon message={parallelError} /> : null}
        </FormSection>

        <FormSection index={3} title="超参数与数据">
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
                <Form.Item name="batch" label={parallel ? '全局 Batch' : 'Batch 大小'} rules={[{ required: true }]}>
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
                  <Select options={['float32', 'float64'].map((v) => ({ value: v, label: v, disabled: parallel && v === 'float64' }))} />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12} md={8}>
                <Form.Item name="conv_backend" label="卷积后端" tooltip={CONV_TOOLTIP}>
                  <ConvSelect options={convChoices} />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12} md={8}>
                <Form.Item name="base_width" label="ResNet 宽度" tooltip="第一阶段通道数，之后每阶段翻倍；默认 64">
                  <InputNumber min={1} placeholder="64" style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12} md={8}>
                <Form.Item name="val_size" label="验证集样本数">
                  <InputNumber min={0} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              <Col xs={24} sm={12} md={8}>
                <Form.Item name="seed" label="初始化种子" tooltip="留空则每次随机初始化">
                  <InputNumber style={{ width: '100%' }} />
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
              <Form.Item name="early_stopping" label="早停" valuePropName="checked">
                <Switch />
              </Form.Item>
              <Form.Item name="export_onnx" label="结束后导出 ONNX" valuePropName="checked">
                <Switch />
              </Form.Item>
            </div>
            {usesNative ? <NativeLibraryNotice status={nativeLibrary} /> : null}
          </Form>
        </FormSection>
      </Panel>

      <aside className="checkout-side">
        {parallel ? (
          <Panel title="拓扑预览" subtitle={`${TRAIN_PARALLEL.find((o) => o.value === parallelKey)?.title} · ${workers} Worker`}>
            <div className="panel-body">
              <EChart option={preview} height={200} />
            </div>
            <div className="panel-footer" style={{ lineHeight: 1.6 }}>
              {distributed === 'ps'
                ? `CPU 参数服务器汇总 ${workers} 个 Worker 的梯度，各 Worker 用相同的平均梯度本地更新。`
                : `${workers} 个对等 Worker 经 ScatterReduce + AllGather 求平均梯度。`}
              每步核对各 Worker 参数一致；停止任务时所有 Worker 在同一步保存断点。
            </div>
          </Panel>
        ) : null}
        <Panel title="配置摘要" subtitle="ResNet18 正式训练">
          <SummaryList
            rows={[
              ['任务', preset?.label],
              ['并行', parallel ? `${TRAIN_PARALLEL.find((o) => o.value === parallelKey)?.title} · ${workers} Worker` : '单进程'],
              ['样本数', maxSamples ? <span className="num" key="m">{maxSamples}</span> : '全量'],
              [
                parallel ? 'Epoch / 全局 Batch' : 'Epoch / Batch',
                <span className="num" key="e">{`${epochs ?? '—'} / ${batch ?? '—'}${shards ? `（${shards.join(' + ')}）` : ''}`}</span>,
              ],
              ['学习率', lr !== undefined ? <span className="num" key="l">{lr}</span> : null],
              ['设备', device],
              ['卷积后端', convChoices.find((o) => o.value === convBackend)?.label],
            ]}
          />
          <SubmitBox
            placeholder="任务名称（可选），如 resnet18_ring2_e5"
            label={label}
            onLabel={setLabel}
            disabled={!preset?.data_ready || !!parallelError}
            loading={submitting}
            onSubmit={submit}
            note="每步写入 metrics.jsonl，控制台每 500 ms 采样整卡硬件；停止任务时先保存最新断点再退出，可用 --resume 续训。"
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
          <div className="page-subtitle">正式训练可选单进程或 PS / Ring AllReduce 数据并行；分布式实验按冻结配置对比通信方案</div>
        </div>
      </div>
      <Tabs
        items={[
          {
            key: 'train',
            label: (
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                <ThunderboltOutlined />
                正式训练
              </span>
            ),
            children: (
              <TrainingForm
                presets={p.single}
                convBackends={p.conv_backends}
                nativeLibrary={p.native_library}
                workerChoices={p.worker_choices}
                maxWorkers={p.max_workers}
              />
            ),
          },
          {
            key: 'dist',
            label: (
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                <BranchesOutlined />
                分布式实验
              </span>
            ),
            children: (
              <DistributedForm
                presets={p.distributed}
                modes={p.modes}
                workerChoices={p.worker_choices}
                maxWorkers={p.max_workers}
                convBackends={p.conv_backends}
                nativeLibrary={p.native_library}
              />
            ),
          },
        ]}
      />
    </div>
  )
}
