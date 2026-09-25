import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Alert, App, Button, Card, Col, Collapse, Empty, Form, Input, InputNumber, Radio, Row, Segmented, Select, Space, Spin, Switch, Tabs, Tag,
} from 'antd'
import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import type { ModeOption, Preset } from '../api'
import EChart from '../components/EChart'
import { topologyOption } from './run/TopologyTab'

const FORM_KEYS = ['device', 'optimizer', 'learning_rate', 'global_batch', 'epochs', 'steps', 'seed', 'timeout',
  'evaluate', 'final_test', 'profile', 'monitor'] as const
const HIDDEN_KEYS = new Set(['task', 'mode', 'transport', 'train_workers', ...FORM_KEYS])

function PresetPicker({ presets, value, onChange }: { presets: Preset[]; value?: string; onChange: (id: string) => void }) {
  return (
    <Radio.Group value={value} onChange={(e) => onChange(e.target.value)} style={{ width: '100%' }}>
      <Row gutter={[12, 12]}>
        {presets.map((p) => (
          <Col xs={24} md={12} xxl={8} key={p.id}>
            <Card size="small" hoverable onClick={() => onChange(p.id)}
              style={{ borderColor: value === p.id ? '#1668dc' : undefined, boxShadow: value === p.id ? '0 0 0 2px rgba(22,104,220,0.15)' : undefined }}>
              <Radio value={p.id}><b>{p.label}</b></Radio>
              <div className="muted" style={{ fontSize: 12, marginTop: 4, minHeight: 36 }}>{p.description}</div>
              {p.data_ready ? <Tag color="success">数据就绪</Tag> : <Tag color="warning">数据未准备</Tag>}
            </Card>
          </Col>
        ))}
      </Row>
    </Radio.Group>
  )
}

function DistributedForm({ presets, modes, workerChoices, maxWorkers }: { presets: Preset[]; modes: ModeOption[]; workerChoices: number[]; maxWorkers: number }) {
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
  const device = (Form.useWatch('device', form) as string | undefined) === 'cpu' ? 'CPU' : 'GPU'
  const batchError = globalBatch !== undefined && globalBatch < workers ? `global_batch（${globalBatch}）必须不小于 worker 数（${workers}）` : null
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
      message.success(`已提交：${job.label}`)
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
    <Row gutter={[16, 16]}>
      <Col span={24}>
        <Card size="small" title="1. 选择任务预设">
          <PresetPicker presets={presets} value={presetId} onChange={setPresetId} />
          {preset && !preset.data_ready ? <Alert style={{ marginTop: 12 }} type="warning" showIcon message={preset.data_hint} /> : null}
        </Card>
      </Col>
      <Col xs={24} xl={14}>
        <Card size="small" title="2. 分布式模式与 Worker">
          <Space orientation="vertical" size={16} style={{ width: '100%' }}>
            <div>
              <div className="metric-label" style={{ marginBottom: 6 }}>模式 / 传输（仅列出框架支持的合法组合）</div>
              <Segmented block value={modeKey} onChange={(v) => setModeKey(String(v))}
                options={modes.map((m) => ({ value: `${m.mode}/${m.transport}`, label: m.label }))} />
            </div>
            <div>
              <div className="metric-label" style={{ marginBottom: 6 }}>{device === 'GPU' ? '训练 Worker 数（所有 Worker 共享本机一张 GPU）' : '训练 Worker 数'}</div>
              <Space>
                <Segmented value={workerChoices.includes(workers) ? workers : 'custom'} disabled={mode === 'single'}
                  onChange={(v) => v !== 'custom' && setWorkers(Number(v))}
                  options={[...workerChoices.map((w) => ({ value: w, label: `${w}` })), { value: 'custom', label: '自定义' }]} />
                <InputNumber min={1} max={maxWorkers} value={workers} disabled={mode === 'single'} onChange={(v) => v && setWorkers(v)} />
              </Space>
            </div>
            {batchError ? <Alert type="error" showIcon message={batchError} /> : null}
            <Form form={form} name="distributed" layout="vertical" requiredMark={false}>
              <Row gutter={12}>
                <Col span={8}><Form.Item name="device" label="设备"><Select options={[{ value: 'cuda', label: 'CUDA' }, { value: 'cpu', label: 'CPU' }]} /></Form.Item></Col>
                <Col span={8}><Form.Item name="optimizer" label="优化器"><Select options={[{ value: 'adam', label: 'Adam' }, { value: 'mbgd', label: 'MBGD' }]} /></Form.Item></Col>
                <Col span={8}><Form.Item name="learning_rate" label="学习率" rules={[{ required: true }]}><InputNumber min={1e-7} step={1e-4} style={{ width: '100%' }} /></Form.Item></Col>
                <Col span={8}><Form.Item name="global_batch" label="全局 batch" rules={[{ required: true }]}><InputNumber min={1} style={{ width: '100%' }} /></Form.Item></Col>
                <Col span={8}><Form.Item name="epochs" label="epoch 数" rules={[{ required: true }]}><InputNumber min={1} style={{ width: '100%' }} /></Form.Item></Col>
                <Col span={8}><Form.Item name="steps" label={preset?.task === 'synthetic' ? '总 step 数' : '每 epoch 最多 batch'} rules={[{ required: true }]}><InputNumber min={1} style={{ width: '100%' }} /></Form.Item></Col>
                <Col span={8}><Form.Item name="seed" label="随机种子"><InputNumber style={{ width: '100%' }} /></Form.Item></Col>
                <Col span={8}><Form.Item name="timeout" label="超时（秒）"><InputNumber min={1} style={{ width: '100%' }} /></Form.Item></Col>
              </Row>
              <Space size={24} wrap>
                <Form.Item name="evaluate" label="验证集评估" valuePropName="checked"><Switch /></Form.Item>
                <Form.Item name="final_test" label="最终测试" valuePropName="checked"><Switch /></Form.Item>
                <Form.Item name="profile" label="CUDA Event 计时" valuePropName="checked"><Switch /></Form.Item>
                <Form.Item name="monitor" label="资源采样" valuePropName="checked"><Switch /></Form.Item>
              </Space>
            </Form>
            <Collapse size="small" items={[{ key: 'adv', label: '高级参数（JSON，会与上面的表单合并）', children: (
              <Input.TextArea value={advanced} onChange={(e) => setAdvanced(e.target.value)} autoSize={{ minRows: 6, maxRows: 18 }}
                style={{ fontFamily: 'Consolas, monospace', fontSize: 12 }} />
            ) }]} />
          </Space>
        </Card>
      </Col>
      <Col xs={24} xl={10}>
        <Card size="small" title="拓扑预览">
          <EChart option={preview} height={260} />
          <div className="muted" style={{ fontSize: 12 }}>
            {mode === 'ps' ? `1 个 CPU PS 进程聚合梯度，${workers} 个 ${device} Worker 在本地更新参数。` :
              mode === 'ring' ? `${workers} 个对等 rank，gRPC 两阶段 ScatterReduce + AllGather，无中心节点。` : '单进程基线，用于与分布式结果对照。'}
          </div>
        </Card>
        <Card size="small" title="3. 提交" style={{ marginTop: 16 }}>
          <Space orientation="vertical" style={{ width: '100%' }}>
            <Input placeholder="任务名称（可选）" value={label} onChange={(e) => setLabel(e.target.value)} />
            <Button type="primary" block size="large" loading={submitting} disabled={!preset || !preset.data_ready || !!batchError} onClick={submit}>
              提交到队列
            </Button>
            <div className="muted" style={{ fontSize: 12 }}>任务按提交顺序串行执行，产物写入 runs/console/。</div>
          </Space>
        </Card>
      </Col>
    </Row>
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

  useEffect(() => {
    if (preset?.args) form.setFieldsValue({ augment: false, graph_opt: false, ...preset.args })
  }, [preset, form])

  const submit = async () => {
    const values = await form.validateFields()
    const args = Object.fromEntries(Object.entries(values).filter(([, v]) => v !== undefined && v !== null && v !== ''))
    setSubmitting(true)
    try {
      const job = await api.submitJob({ type: 'single', args, label: label || undefined })
      message.success(`已提交：${job.label}`)
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
      navigate(`/run/console/${job.id}`)
    } catch (e) {
      message.error((e as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Row gutter={[16, 16]}>
      <Col span={24}>
        <Card size="small" title="1. 选择预设（apps.train.train_myflows_donkey）">
          <PresetPicker presets={presets} value={presetId} onChange={setPresetId} />
        </Card>
      </Col>
      <Col xs={24} xl={14}>
        <Card size="small" title="2. 训练参数">
          <Form form={form} name="single" layout="vertical" requiredMark={false}>
            <Row gutter={12}>
              <Col span={8}><Form.Item name="max_samples" label="样本数（0 = 全部）"><InputNumber min={0} style={{ width: '100%' }} /></Form.Item></Col>
              <Col span={8}><Form.Item name="epochs" label="epoch 数" rules={[{ required: true }]}><InputNumber min={1} style={{ width: '100%' }} /></Form.Item></Col>
              <Col span={8}><Form.Item name="batch" label="batch" rules={[{ required: true }]}><InputNumber min={1} style={{ width: '100%' }} /></Form.Item></Col>
              <Col span={8}><Form.Item name="lr" label="学习率"><InputNumber min={1e-7} step={1e-4} style={{ width: '100%' }} /></Form.Item></Col>
              <Col span={8}><Form.Item name="device" label="设备"><Select options={['auto', 'cuda', 'cpu'].map((v) => ({ value: v, label: v }))} /></Form.Item></Col>
              <Col span={8}><Form.Item name="dtype" label="精度"><Select options={['float32', 'float64'].map((v) => ({ value: v, label: v }))} /></Form.Item></Col>
              <Col span={8}><Form.Item name="val_size" label="验证集条数"><InputNumber min={0} style={{ width: '100%' }} /></Form.Item></Col>
              <Col span={8}><Form.Item name="checkpoint_every" label="每多少 step 存断点"><InputNumber min={0} style={{ width: '100%' }} /></Form.Item></Col>
              <Col span={8}><Form.Item name="tb_log_interval" label="TensorBoard 记录间隔"><InputNumber min={1} style={{ width: '100%' }} /></Form.Item></Col>
            </Row>
            <Space size={24}>
              <Form.Item name="augment" label="数据增强" valuePropName="checked"><Switch /></Form.Item>
              <Form.Item name="graph_opt" label="构图优化" valuePropName="checked"><Switch /></Form.Item>
            </Space>
          </Form>
        </Card>
      </Col>
      <Col xs={24} xl={10}>
        <Card size="small" title="3. 提交">
          <Space orientation="vertical" style={{ width: '100%' }}>
            <Alert type="info" showIcon message="单进程训练每步写 metrics.jsonl，控制台以 500 ms 间隔记录 GPU/CPU；停止时先写 stop 文件，训练保存断点后退出。" />
            <Input placeholder="任务名称（可选）" value={label} onChange={(e) => setLabel(e.target.value)} />
            <Button type="primary" block size="large" loading={submitting} disabled={!preset?.data_ready} onClick={submit}>提交到队列</Button>
          </Space>
        </Card>
      </Col>
    </Row>
  )
}

export default function NewJob() {
  const presets = useQuery({ queryKey: ['presets'], queryFn: api.presets })
  if (presets.isLoading) return <Spin style={{ display: 'block', marginTop: 80 }} />
  if (!presets.data) return <Empty description="无法加载预设" />
  const p = presets.data
  return (
    <div>
      <div className="page-title"><h2>新建训练</h2></div>
      <Tabs items={[
        { key: 'dist', label: '分布式训练（PS / Ring）', children: <DistributedForm presets={p.distributed} modes={p.modes} workerChoices={p.worker_choices} maxWorkers={p.max_workers} /> },
        { key: 'single', label: '单进程训练（apps.train）', children: <SingleForm presets={p.single} /> },
      ]} />
    </div>
  )
}
