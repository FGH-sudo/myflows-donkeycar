import { ExperimentOutlined, RocketOutlined } from '@ant-design/icons'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, App, Button, Checkbox, Col, Collapse, Form, Input, InputNumber, Row, Segmented, Select, Space, Spin, Switch, Tooltip } from 'antd'
import type { ReactNode } from 'react'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { autopilot } from '../../api'
import type { CampaignSpec } from '../../api'
import EmptyState from '../../components/EmptyState'
import { Panel } from '../../components/Panel'
import { campaignPath } from '../../format'

const CONV_BACKENDS = [
  { value: 'cuda_native_register', label: '原生 · 寄存器分块（默认）' },
  { value: 'cuda_native_cublas', label: '原生 · cuBLAS' },
  { value: 'cupy', label: 'CuPy' },
  { value: 'numpy', label: 'NumPy（CPU）' },
]

function Section({ index, title, description, children }: { index: number; title: string; description?: ReactNode; children: ReactNode }) {
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

const toNumbers = (values: (string | number)[] | undefined) =>
  Array.from(new Set((values ?? []).map((v) => Number(v)).filter((v) => Number.isFinite(v)))).sort((a, b) => a - b)

type FormValues = CampaignSpec & {
  space_lr: (string | number)[]
  space_batch: (string | number)[]
  space_weight_decay?: (string | number)[]
  space_dropout?: (string | number)[]
  space_augment?: string[]
  space_freeze_stages?: number[]
  seeds: (string | number)[]
}

// E2 transfer learning: torchvision-compatible structures and the number of frozen stages (0..5).
const ARCH_OPTIONS = [
  { value: 'resnet18', label: 'ResNet18' },
  { value: 'mobilenet_v2', label: 'MobileNetV2' },
  { value: 'alexnet', label: 'AlexNet' },
]
const FREEZE_OPTIONS = [0, 1, 2, 3, 4, 5].map((v) => ({ value: v, label: v === 0 ? '0（全部微调）' : `${v}` }))

const AUGMENT_OPTIONS = [
  { value: 'off', label: '不增强' },
  { value: 'on', label: '增强' },
]

export default function NewCampaign() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { message, modal } = App.useApp()
  const defaults = useQuery({ queryKey: ['autopilot', 'defaults'], queryFn: autopilot.defaults })
  const [form] = Form.useForm<FormValues>()
  const [submitting, setSubmitting] = useState(false)
  const mode = Form.useWatch('mode', form)
  const maxSamples = Form.useWatch(['fixed', 'max_samples'], form)
  const llmEnabled = Form.useWatch(['llm', 'enabled'], form)
  const strategyName = Form.useWatch(['strategy', 'name'], form)
  const splitKind = Form.useWatch(['fixed', 'split_kind'], form)
  const pretrained = Form.useWatch(['fixed', 'pretrained'], form)
  const arch = Form.useWatch(['fixed', 'arch'], form)
  // preserve: the rollback fields live in a collapsed panel that is not mounted until opened
  const rollbackEnabled = Form.useWatch(['rollback', 'enabled'], { form, preserve: true })

  useEffect(() => {
    if (!defaults.data) return
    const spec = defaults.data.spec
    form.setFieldsValue({
      ...spec,
      campaign_id: undefined as unknown as null,
      space_lr: spec.space.lr as number[],
      space_batch: spec.space.batch as number[],
      space_weight_decay: (spec.space.weight_decay as number[] | undefined) ?? [],
      space_dropout: (spec.space.dropout as number[] | undefined) ?? [],
      space_augment: ((spec.space.augment as boolean[] | undefined) ?? []).map((a) => (a ? 'on' : 'off')),
      space_freeze_stages: (spec.space.freeze_stages as number[] | undefined) ?? [],
      seeds: spec.baseline.seeds,
      llm: { ...spec.llm, enabled: spec.llm.enabled && defaults.data.llm_available },
    })
  }, [defaults.data, form])

  if (defaults.isLoading) return <Spin style={{ display: 'block', marginTop: 80 }} />
  if (!defaults.data) {
    return (
      <Panel>
        <EmptyState title="无法加载默认配置" description="请确认控制台后端正在运行" />
      </Panel>
    )
  }
  const d = defaults.data
  const task = d.tasks[0]

  const build = (): Partial<CampaignSpec> => {
    const v = form.getFieldsValue(true) as FormValues
    const space: CampaignSpec['space'] = { lr: toNumbers(v.space_lr), batch: toNumbers(v.space_batch) }
    if (v.strategy?.name === 'tpe') {
      // Regularisation joins the search only with TPE (E3); empty lists leave the key out of the space.
      const weightDecay = toNumbers(v.space_weight_decay).filter((x) => x > 0)
      const dropout = toNumbers(v.space_dropout).filter((x) => x > 0 && x < 1)
      if (weightDecay.length) space.weight_decay = weightDecay
      if (dropout.length) space.dropout = dropout
      if (v.space_augment?.length) space.augment = Array.from(new Set(v.space_augment)).map((a) => a === 'on')
    }
    // freeze_stages is searchable only with pretrained weights (the server re-checks the capability).
    if (v.fixed?.pretrained && v.space_freeze_stages?.length) space.freeze_stages = toNumbers(v.space_freeze_stages)
    return {
      ...(v.campaign_id ? { campaign_id: v.campaign_id } : {}),
      title: v.title ?? '',
      mode: v.mode,
      task: v.task,
      objective: v.objective,
      baseline: { ...v.baseline, seeds: toNumbers(v.seeds) },
      fixed: Object.fromEntries(
        Object.entries(v.fixed).filter(([k, val]) => k !== 'split_file' && val !== undefined && val !== null && val !== ''),
      ),
      space,
      strategy: v.strategy,
      budget: v.budget,
      online: v.online,
      specialists: v.specialists,
      llm: v.llm,
      rollback: v.rollback,
    }
  }

  const submit = async (start: boolean) => {
    try {
      await form.validateFields()
    } catch {
      return
    }
    const spec = build()
    const run = async () => {
      setSubmitting(true)
      try {
        const created = await autopilot.create(spec)
        if (start) await autopilot.action(created.campaign_id, 'start')
        message.success(start ? 'campaign 已创建并启动' : 'campaign 已创建')
        queryClient.invalidateQueries({ queryKey: ['autopilot'] })
        navigate(campaignPath(created.campaign_id))
      } catch (e) {
        message.error((e as Error).message)
      } finally {
        setSubmitting(false)
      }
    }
    if (start && (!spec.fixed?.max_samples || Number(spec.fixed.max_samples) > 5000)) {
      modal.confirm({
        title: '全量数据会长时间占用 GPU',
        content: '开发与验证请用小规模数据；跑全量之前先确认机器已经空出来。确定现在启动？',
        okText: '确认启动',
        onOk: run,
      })
    } else {
      await run()
    }
  }

  const switchMode = (value: string) => {
    const spec = value === 'formal' ? d.formal : d.spec
    // Data, split, test set and rollback follow the mode (formal = full data; starting it still asks first).
    const { max_samples, val_size, test_size, split_kind, block_size } = spec.fixed
    form.setFieldsValue({
      mode: value as CampaignSpec['mode'],
      seeds: spec.baseline.seeds,
      fixed: { ...form.getFieldValue('fixed'), max_samples, val_size, test_size, split_kind, block_size },
      rollback: spec.rollback,
    })
  }

  return (
    <div>
      <div className="page-title">
        <div>
          <h2>新建 campaign</h2>
          <div className="page-subtitle">规则内核负责调度与判断；LLM 顾问可选，它发起的启动训练操作一律需要人工确认</div>
        </div>
      </div>
      <div className="checkout">
        <Panel>
          <Form form={form} layout="vertical" requiredMark={false}>
            <Section index={1} title="目标与模式" description="联调用 2 个 seed，正式用 3 个；正式 campaign 前需要按时间段划分并留出测试集">
              <Row gutter={16}>
                <Col xs={24} md={12}>
                  <Form.Item name="title" label="标题">
                    <Input placeholder="例如：ResNet18 lr × batch 小规模搜索" />
                  </Form.Item>
                </Col>
                <Col xs={24} md={12}>
                  <Form.Item name="campaign_id" label="campaign_id（可选）" tooltip="留空则按任务与时间自动生成；小写字母、数字、- 与 _">
                    <Input placeholder="donkey-r18-lr-batch-001" />
                  </Form.Item>
                </Col>
                <Col xs={24} md={8}>
                  <Form.Item name="mode" label="模式">
                    <Segmented
                      block
                      options={Object.entries(d.modes).map(([value, label]) => ({ value, label }))}
                      onChange={(v) => switchMode(String(v))}
                    />
                  </Form.Item>
                </Col>
                <Col xs={24} md={8}>
                  <Form.Item name="task" label="任务">
                    <Select options={d.tasks.map((t) => ({ value: t.task, label: t.label }))} />
                  </Form.Item>
                </Col>
                <Col xs={24} md={8}>
                  <Form.Item name={['objective', 'metric']} label="目标指标">
                    <Select options={Object.entries(task.metrics).map(([m, dir]) => ({ value: m, label: `${m}（${dir === 'minimize' ? '越低越好' : '越高越好'}）` }))} />
                  </Form.Item>
                </Col>
              </Row>
            </Section>

            <Section
              index={2}
              title="数据与固定参数"
              description="所有 trial 用同一个固定划分的验证集；样本数 0 表示全量；测试集只在 campaign 结束时评估一次"
            >
              <Row gutter={16}>
                <Col xs={12} md={6}>
                  <Form.Item name={['fixed', 'max_samples']} label="样本数">
                    <InputNumber min={0} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col xs={12} md={6}>
                  <Form.Item name={['fixed', 'epochs']} label="Epoch">
                    <InputNumber min={1} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col xs={12} md={6}>
                  <Form.Item name={['fixed', 'val_size']} label="验证集条数">
                    <InputNumber min={1} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col xs={12} md={6}>
                  <Form.Item name={['fixed', 'split_seed']} label="划分种子">
                    <InputNumber min={0} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col xs={24} md={10}>
                  <Form.Item
                    name={['fixed', 'split_kind']}
                    label="划分方式"
                    tooltip="按时间段：以连续帧为块抽取验证/测试集，并在块两侧留出间隔，避免相邻帧泄漏"
                  >
                    <Select
                      options={[
                        { value: 'random', label: '随机抽样' },
                        { value: 'time_block', label: '按时间段（正式 campaign）' },
                      ]}
                    />
                  </Form.Item>
                </Col>
                <Col xs={12} md={7}>
                  <Form.Item name={['fixed', 'block_size']} label="时间块帧数">
                    <InputNumber min={1} style={{ width: '100%' }} disabled={splitKind !== 'time_block'} />
                  </Form.Item>
                </Col>
                <Col xs={12} md={7}>
                  <Form.Item name={['fixed', 'test_size']} label="测试集条数" tooltip="0 表示不留测试集；正式 campaign 必须留">
                    <InputNumber min={0} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col xs={12} md={6}>
                  <Form.Item name={['fixed', 'device']} label="设备">
                    <Select options={['cuda', 'auto', 'cpu'].map((v) => ({ value: v, label: v }))} />
                  </Form.Item>
                </Col>
                <Col xs={24} md={10}>
                  <Form.Item name={['fixed', 'conv_backend']} label="卷积后端" tooltip="固定为寄存器分块；Operator Optimization 按吞吐复核">
                    <Select options={CONV_BACKENDS} />
                  </Form.Item>
                </Col>
                <Col xs={12} md={4}>
                  <Form.Item name={['fixed', 'base_width']} label="ResNet 宽度">
                    <InputNumber min={1} style={{ width: '100%' }} disabled={Boolean(pretrained) || Boolean(arch && arch !== 'resnet18')} />
                  </Form.Item>
                </Col>
                <Col xs={12} md={4}>
                  <Form.Item name={['fixed', 'stem']} label="Stem">
                    <Select
                      options={['cifar', 'imagenet'].map((v) => ({ value: v, label: v }))}
                      disabled={Boolean(pretrained) || Boolean(arch && arch !== 'resnet18')}
                    />
                  </Form.Item>
                </Col>
                <Col xs={12} md={6}>
                  <Form.Item
                    name={['fixed', 'arch']}
                    label="网络结构"
                    tooltip="MobileNetV2、AlexNet 以及带预训练权重的 ResNet18 与 torchvision 结构逐层一致（E2）"
                  >
                    <Select options={ARCH_OPTIONS} placeholder="ResNet18" allowClear />
                  </Form.Item>
                </Col>
                <Col xs={12} md={10}>
                  <Form.Item
                    name={['fixed', 'pretrained']}
                    label="预训练权重"
                    tooltip="PyTorch 官方 ImageNet 权重（tools/convert_torchvision_weights.py 转换后）；载入主干、替换回归头，输入按 ImageNet 均值/标准差归一化"
                  >
                    <Select
                      allowClear
                      placeholder="不使用（随机初始化）"
                      options={[{ value: 'imagenet', label: 'ImageNet（torchvision 官方）' }]}
                      disabled={d.specialists.transfer && !d.specialists.transfer.available}
                    />
                  </Form.Item>
                </Col>
                <Col xs={12} md={8}>
                  <Form.Item name={['fixed', 'image_size']} label="输入尺寸 HxW" tooltip="留空则与首帧一致（DonkeyCar 为 120x160）">
                    <Input placeholder="例如 224x224" allowClear />
                  </Form.Item>
                </Col>
              </Row>
              {!maxSamples || Number(maxSamples) > 5000 ? (
                <Alert type="warning" showIcon message="全量或大规模数据：开发与验证请用小规模数据，跑全量之前先确认机器已空出。" />
              ) : null}
            </Section>

            <Section index={3} title="人工基线" description="基线 × 每个 seed 各跑一次，得到基线与噪声水平；提升小于基线 seed 间标准差时报告写“无显著差别”">
              <Row gutter={16}>
                <Col xs={12} md={6}>
                  <Form.Item name={['baseline', 'lr']} label="lr">
                    <InputNumber min={1e-7} step={1e-4} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col xs={12} md={6}>
                  <Form.Item name={['baseline', 'batch']} label="batch">
                    <InputNumber min={1} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col xs={12} md={6}>
                  <Form.Item name={['baseline', 'num_workers']} label="num_workers">
                    <InputNumber min={0} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col xs={12} md={6}>
                  <Form.Item name="seeds" label={`seed（${mode === 'formal' ? '正式 3 个' : '联调 2 个'}）`}>
                    <Select mode="tags" tokenSeparators={[',', ' ']} />
                  </Form.Item>
                </Col>
              </Row>
            </Section>

            <Section
              index={4}
              title="搜索空间与策略"
              description="精度搜 lr × batch，TPE 策略下再加入 weight_decay、dropout 和数据增强；num_workers、卷积后端和训练 worker 数由吞吐量决定"
            >
              <Row gutter={16}>
                <Col xs={24} md={12}>
                  <Form.Item name="space_lr" label="lr 候选">
                    <Select mode="tags" tokenSeparators={[',', ' ']} />
                  </Form.Item>
                </Col>
                <Col xs={24} md={12}>
                  <Form.Item name="space_batch" label="batch 候选">
                    <Select mode="tags" tokenSeparators={[',', ' ']} />
                  </Form.Item>
                </Col>
                <Col xs={12} md={8}>
                  <Form.Item name={['strategy', 'name']} label="策略">
                    <Select
                      options={d.strategies.map((s) => ({
                        value: s,
                        label: s === 'candidate_list' ? '分层抽样（CandidateList）' : 'TPE（Optuna，E3）',
                      }))}
                    />
                  </Form.Item>
                </Col>
                <Col xs={12} md={8}>
                  <Form.Item name={['strategy', 'max_candidates']} label="最多候选数">
                    <InputNumber min={1} max={60} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col xs={12} md={8}>
                  <Form.Item name={['strategy', 'seed']} label="抽样种子">
                    <InputNumber min={0} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
              </Row>
              {pretrained ? (
                <Row gutter={16}>
                  <Col xs={24} md={12}>
                    <Form.Item
                      name="space_freeze_stages"
                      label="冻结阶段数候选"
                      tooltip="冻结前 N 个阶段（参数不训练，BN 固定 running statistics）；Transfer Learning 子 Agent 围绕最佳 trial 比较相邻的冻结阶段数"
                    >
                      <Select mode="multiple" options={FREEZE_OPTIONS} placeholder="留空则不搜（全部微调）" />
                    </Form.Item>
                  </Col>
                </Row>
              ) : null}
              {strategyName === 'tpe' ? (
                <Row gutter={16}>
                  <Col xs={24} md={8}>
                    <Form.Item
                      name="space_weight_decay"
                      label="weight_decay 范围"
                      tooltip="TPE 在最小值与最大值之间按对数尺度取值；与 dropout 同一时间只启用一种正则"
                    >
                      <Select mode="tags" tokenSeparators={[',', ' ']} placeholder="例如 0.0001, 0.01（留空则不搜）" />
                    </Form.Item>
                  </Col>
                  <Col xs={24} md={8}>
                    <Form.Item name="space_dropout" label="dropout 范围">
                      <Select mode="tags" tokenSeparators={[',', ' ']} placeholder="例如 0.1, 0.5（留空则不搜）" />
                    </Form.Item>
                  </Col>
                  <Col xs={24} md={8}>
                    <Form.Item name="space_augment" label="数据增强">
                      <Select mode="multiple" options={AUGMENT_OPTIONS} placeholder="留空则不搜" />
                    </Form.Item>
                  </Col>
                </Row>
              ) : null}
            </Section>

            <Section index={5} title="预算" description="总时长只计 trial 的实际运行时间（获得 GPU 锁之后），不计排队等待">
              <Row gutter={16}>
                <Col xs={12} md={6}>
                  <Form.Item name={['budget', 'max_trials']} label="最多 trial 数">
                    <InputNumber min={1} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col xs={12} md={6}>
                  <Form.Item name={['budget', 'trial_timeout_s']} label="单 trial 上限（秒）">
                    <InputNumber min={10} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col xs={12} md={6}>
                  <Form.Item name={['budget', 'wall_time_s']} label="总时长（秒）">
                    <InputNumber min={60} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col xs={12} md={6}>
                  <Form.Item name={['budget', 'max_retries']} label="基础设施失败重试">
                    <InputNumber min={0} max={3} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
              </Row>
              <Collapse
                ghost
                size="small"
                items={[
                  {
                    key: 'online',
                    label: '在线检查阈值',
                    children: (
                      <Row gutter={16}>
                        <Col xs={12} md={6}>
                          <Form.Item name={['online', 'hang_timeout_s']} label="卡死判定（秒）">
                            <InputNumber min={5} style={{ width: '100%' }} />
                          </Form.Item>
                        </Col>
                        <Col xs={12} md={6}>
                          <Form.Item name={['online', 'divergence_windows']} label="发散：连续上升窗口">
                            <InputNumber min={1} style={{ width: '100%' }} />
                          </Form.Item>
                        </Col>
                        <Col xs={12} md={6}>
                          <Form.Item name={['online', 'divergence_factor']} label="发散：超过初始值倍数">
                            <InputNumber min={1} step={0.5} style={{ width: '100%' }} />
                          </Form.Item>
                        </Col>
                        <Col xs={12} md={6}>
                          <Form.Item
                            name={['online', 'explosion_factor']}
                            label="爆炸：超过首步 loss 倍数"
                            tooltip="loss 的 EMA 连续多个窗口都超过首步 loss 的这个倍数，也判为发散；0 表示关闭"
                          >
                            <InputNumber min={0} step={10} style={{ width: '100%' }} />
                          </Form.Item>
                        </Col>
                        <Col xs={12} md={6}>
                          <Form.Item name={['online', 'gpu_idle_pct']} label="GPU 空闲阈值（%）">
                            <InputNumber min={0} max={100} style={{ width: '100%' }} />
                          </Form.Item>
                        </Col>
                        <Col xs={12} md={6}>
                          <Form.Item name={['budget', 'max_gpu_memory_mb']} label="显存预算（MB，可空）">
                            <InputNumber min={1} style={{ width: '100%' }} />
                          </Form.Item>
                        </Col>
                      </Row>
                    ),
                  },
                  {
                    key: 'rollback',
                    forceRender: true,
                    label: `异常回滚（E3）${rollbackEnabled ? '：已开启' : ''}`,
                    children: (
                      <Row gutter={16}>
                        <Col xs={12} md={6}>
                          <Form.Item
                            name={['rollback', 'enabled']}
                            label="精确回滚"
                            valuePropName="checked"
                            tooltip="NaN 或发散时，从失败前最近的健康 TrainingState 逐位续训，并降低 lr；数据并行 trial 不回滚"
                          >
                            <Switch />
                          </Form.Item>
                        </Col>
                        <Col xs={12} md={6}>
                          <Form.Item name={['rollback', 'max_rollbacks']} label="每个 trial 最多回滚">
                            <InputNumber min={0} max={5} style={{ width: '100%' }} disabled={!rollbackEnabled} />
                          </Form.Item>
                        </Col>
                        <Col xs={12} md={6}>
                          <Form.Item name={['rollback', 'lr_factor']} label="回滚后 lr 乘以">
                            <InputNumber min={0.01} max={0.99} step={0.1} style={{ width: '100%' }} disabled={!rollbackEnabled} />
                          </Form.Item>
                        </Col>
                        <Col xs={12} md={6}>
                          <Form.Item name={['rollback', 'state_every']} label="TrainingState 间隔（步）">
                            <InputNumber min={1} style={{ width: '100%' }} disabled={!rollbackEnabled} />
                          </Form.Item>
                        </Col>
                        <Col xs={12} md={6}>
                          <Form.Item name={['rollback', 'oom_half_batch']} label="OOM 后派生半 batch" valuePropName="checked">
                            <Switch disabled={!rollbackEnabled} />
                          </Form.Item>
                        </Col>
                      </Row>
                    ),
                  },
                ]}
              />
            </Section>

            <Section index={6} title="子 Agent 与 LLM" description="子 Agent 是确定性规则；LLM 顾问在决策点读取证据并提出建议，每个 campaign 最多调用 30 次">
              <Form.Item name="specialists" label="专业子 Agent">
                <Checkbox.Group>
                  <Space wrap>
                    {Object.entries(d.specialists).map(([name, s]) => (
                      <Tooltip key={name} title={s.available ? undefined : s.reason}>
                        <Checkbox value={name} disabled={name === 'transfer' && (!s.available || !pretrained)}>
                          {s.label}
                          {s.available ? '' : '（不可用）'}
                        </Checkbox>
                      </Tooltip>
                    ))}
                  </Space>
                </Checkbox.Group>
              </Form.Item>
              <Row gutter={16} align="middle">
                <Col xs={12} md={6}>
                  <Form.Item name={['llm', 'enabled']} label="LLM 顾问" valuePropName="checked">
                    <Switch />
                  </Form.Item>
                </Col>
                <Col xs={12} md={6}>
                  <Form.Item name={['llm', 'max_calls']} label="后台调用上限">
                    <InputNumber min={0} max={30} style={{ width: '100%' }} disabled={!llmEnabled} />
                  </Form.Item>
                </Col>
                <Col xs={24} md={12}>
                  {!d.llm_available ? (
                    <Alert type="info" showIcon message="LLM 层当前不可用（未配置 Key 或 dsh 未就绪）：开启后 campaign 仍按规则策略运行。" />
                  ) : null}
                </Col>
              </Row>
            </Section>
          </Form>
        </Panel>
        <aside className="checkout-side">
          <Panel title="流程" subtitle="规则内核 · 单 GPU 串行">
            <div className="panel-body" style={{ lineHeight: 1.7 }}>
              <ol style={{ paddingLeft: 18, margin: 0 }}>
                <li>baseline：人工基线 × 每个 seed</li>
                <li>explore：分层抽样或 TPE 提出候选（预算用到 70% 为止）</li>
                <li>exploit：GPU Monitor 等子 Agent 按证据派生 trial</li>
                <li>confirm：前 1～2 名换新 seed 独立重跑，与基线配对比较</li>
              </ol>
            </div>
            <div className="submit-box">
              <Button type="primary" block size="large" icon={<RocketOutlined />} loading={submitting} onClick={() => submit(true)}>
                创建并启动
              </Button>
              <Button block icon={<ExperimentOutlined />} disabled={submitting} onClick={() => submit(false)}>
                只创建（草稿）
              </Button>
              <div className="submit-note">
                trial 交给控制台任务队列，与手动提交的训练串行执行；终端里启动的训练也经同一把 GPU 锁排队。
              </div>
            </div>
          </Panel>
        </aside>
      </div>
    </div>
  )
}
