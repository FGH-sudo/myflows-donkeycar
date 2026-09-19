import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const pptxgen = require('pptxgenjs');
const JSZip = require('jszip');

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const reportRoot = path.join(repo, 'docs', 'experiments', 'semester_2026_fall', 'distributed_gpu', '20260919_optimized');
const outPath = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.join(reportRoot, '第四周汇报_PS_Ring_分布式训练.pptx');

function readJson(name) {
  return JSON.parse(fs.readFileSync(path.join(reportRoot, name), 'utf8'));
}

function readCsv(name) {
  const lines = fs.readFileSync(path.join(reportRoot, name), 'utf8').trim().split(/\r?\n/);
  const headers = lines.shift().split(',');
  return lines.map(line => {
    const values = line.split(',');
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? '']));
  });
}

const rows = readCsv('comparison.csv');
const summary = readJson('delivery_summary.json');
const byConfig = new Map(rows.map(row => [`${row.task}/${row.config_id}`, row]));
const get = (task, config) => byConfig.get(`${task}/${config}`);
const num = (row, field) => Number(row[field]);
const mb = bytes => (Number(bytes) / 1_000_000).toFixed(1);

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'testmyflow';
pptx.company = 'testmyflow';
pptx.subject = '第四周汇报：分布式训练与 GPU 实现';
pptx.title = '第四周汇报：MNIST、DonkeyCar、PS 与 Ring AllReduce';
pptx.lang = 'zh-CN';
pptx.theme = {
  headFontFace: 'Microsoft YaHei UI',
  bodyFontFace: 'Microsoft YaHei UI',
  lang: 'zh-CN',
};

const C = {
  navy: '102A43',
  blue: '1667A8',
  cyan: '0E9F9A',
  orange: 'E07A3F',
  gold: 'D6A84F',
  ink: '1F2933',
  muted: '52606D',
  light: 'F5F8FB',
  paleBlue: 'E8F1F8',
  paleCyan: 'E5F5F3',
  paleOrange: 'FFF0E8',
  grid: 'D9E2EC',
  white: 'FFFFFF',
  red: 'B42318',
};
const FONT = 'Microsoft YaHei UI';
const W = 13.333;
const H = 7.5;
const S = pptx.ShapeType;

function addText(slide, text, opts = {}) {
  slide.addText(text, {
    fontFace: FONT,
    color: C.ink,
    margin: 0,
    breakLine: false,
    fit: 'shrink',
    ...opts,
  });
}

function addBase(slide, title, section = '第四周汇报') {
  slide.background = { color: C.light };
  slide.addShape(S.rect, { x: 0, y: 0, w: W, h: 0.22, fill: { color: C.blue }, line: { color: C.blue } });
  addText(slide, section.toUpperCase(), { x: 0.62, y: 0.34, w: 2.2, h: 0.24, fontSize: 10, bold: true, color: C.cyan, charSpacing: 1.2 });
  addText(slide, title, { x: 0.62, y: 0.62, w: 12.05, h: 0.48, fontSize: 27, bold: true, color: C.navy });
  slide.addShape(S.line, { x: 0.62, y: 1.22, w: 12.08, h: 0, line: { color: C.grid, pt: 1 } });
  addText(slide, 'testmyflow · 单机多进程 · RTX 4060 Laptop', { x: 0.62, y: 7.16, w: 5.4, h: 0.16, fontSize: 8.5, color: C.muted });
}

function addNotes(slide, notes) {
  slide.addNotes(notes);
}

function callout(slide, x, y, w, h, value, label, fill = C.white, valueColor = C.navy) {
  slide.addShape(S.roundRect, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: C.grid, pt: 0.8 } });
  addText(slide, value, { x: x + 0.16, y: y + 0.16, w: w - 0.32, h: 0.38, fontSize: 24, bold: true, color: valueColor, align: 'center' });
  addText(slide, label, { x: x + 0.12, y: y + 0.62, w: w - 0.24, h: 0.28, fontSize: 10.5, color: C.muted, align: 'center' });
}

function cell(text, fill, color = C.ink, bold = false, align = 'center') {
  return { text: String(text), options: { fill: { color: fill }, color, bold, align, valign: 'middle', margin: 0.06 } };
}

function table(slide, data, opts = {}) {
  const header = data[0];
  const body = data.slice(1);
  const styled = [header.map(v => cell(v, C.navy, C.white, true)), ...body.map((row, i) => row.map(v => cell(v, i % 2 ? C.white : C.paleBlue)))];
  slide.addTable(styled, {
    x: opts.x, y: opts.y, w: opts.w, h: opts.h,
    colW: opts.colW,
    rowH: opts.rowH,
    border: { pt: 0.6, color: C.grid },
    fontFace: FONT,
    fontSize: opts.fontSize ?? 11,
    margin: 0.06,
    autoFit: false,
  });
}

function arrow(slide, x1, y1, x2, y2, color = C.blue) {
  // DrawingML extents must be non-negative, including for reverse arrows.
  slide.addShape(S.line, {
    x: Math.min(x1, x2), y: Math.min(y1, y2),
    w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipH: x2 < x1, flipV: y2 < y1,
    line: { color, pt: 1.6, endArrowType: 'triangle' },
  });
}

function node(slide, x, y, w, h, text, fill, color = C.ink) {
  slide.addShape(S.roundRect, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill, pt: 0.8 } });
  addText(slide, text, { x: x + 0.08, y: y + 0.11, w: w - 0.16, h: h - 0.22, fontSize: 13, bold: true, color, align: 'center', valign: 'middle', breakLine: true });
}

// 1. Cover
{
  const slide = pptx.addSlide();
  slide.background = { color: C.navy };
  slide.addShape(S.rect, { x: 0, y: 0, w: 0.22, h: H, fill: { color: C.cyan }, line: { color: C.cyan } });
  addText(slide, '第四周汇报', { x: 0.72, y: 0.78, w: 4.0, h: 0.35, fontSize: 18, bold: true, color: '8DE0D9', charSpacing: 2 });
  addText(slide, '分布式训练与 GPU 实现', { x: 0.72, y: 1.35, w: 7.2, h: 0.72, fontSize: 34, bold: true, color: C.white });
  addText(slide, 'MNIST MLP · DonkeyCar ResNet18 · Parameter Server · Ring AllReduce', { x: 0.76, y: 2.25, w: 7.8, h: 0.35, fontSize: 15, color: 'D9E2EC' });
  addText(slide, '同条件单机模拟：记录计算、通信、消息量、准确率与加速比', { x: 0.76, y: 2.82, w: 6.5, h: 0.3, fontSize: 12.5, color: 'B8C7D9' });
  addText(slide, '2026-09-19', { x: 0.76, y: 6.65, w: 2.0, h: 0.22, fontSize: 11, color: 'B8C7D9' });

  // Editable topology motif
  slide.addShape(S.ellipse, { x: 9.42, y: 2.45, w: 1.6, h: 1.6, fill: { color: C.cyan }, line: { color: C.cyan } });
  addText(slide, '单机\nGPU', { x: 9.67, y: 2.88, w: 1.1, h: 0.55, fontSize: 18, bold: true, color: C.navy, align: 'center', breakLine: true });
  const coverNodes = [[8.0, 1.4, 'PS'], [11.0, 1.4, 'Ring'], [8.0, 4.9, 'Worker'], [11.0, 4.9, 'Worker']];
  for (const [x, y, text] of coverNodes) {
    slide.addShape(S.ellipse, { x, y, w: 1.05, h: 1.05, fill: { color: '284B63' }, line: { color: '5BC5BE', pt: 1.1 } });
    addText(slide, text, { x, y: y + 0.37, w: 1.05, h: 0.18, fontSize: 12, bold: true, color: C.white, align: 'center' });
    const dx = x + 0.525 - 10.22;
    const dy = y + 0.525 - 3.25;
    const distance = Math.hypot(dx, dy);
    arrow(slide, 10.22 + 0.84 * dx / distance, 3.25 + 0.84 * dy / distance,
      x + 0.525 - 0.57 * dx / distance, y + 0.525 - 0.57 * dy / distance, '5BC5BE');
  }
  addNotes(slide, '本页为汇报封面。课程依据：深度学习框架-16 综合项目III.pdf 第 4 次课主题为分布式计算、GPU 编程实现以及 CUDA Nsight 工具。');
}

// 2. Requirements and scope
{
  const slide = pptx.addSlide();
  addBase(slide, '课程要求与本次完成范围');
  addText(slide, '三份课件的要求被拆成“系统、测量、对照、解释”四类证据。', { x: 0.7, y: 1.43, w: 7.6, h: 0.28, fontSize: 15, color: C.muted });
  const req = [
    ['系统', 'PS：1 个 PS + 2/4 个 Worker + Launcher + Monitor；Ring：Split / ScatterReduce / AllGather'],
    ['任务', 'MNIST + MLP 先跑通，再做 CNN / DonkeyCar 道路任务'],
    ['测量', '每个 epoch 分开记录总时间、计算时间、通信时间、消息体积、准确率 / MAE'],
    ['解释', '加速比与线性理想值对照，指出通信、同步、数据与单卡资源瓶颈'],
  ];
  table(slide, [['课件要求', '落实方式'], ...req], { x: 0.7, y: 1.88, w: 7.5, h: 3.25, colW: [1.2, 6.3], rowH: 0.72, fontSize: 12 });
  callout(slide, 8.75, 1.55, 1.75, 1.28, '56', '正式档案', C.paleBlue, C.blue);
  callout(slide, 10.7, 1.55, 1.75, 1.28, '42', '性能重复', C.paleCyan, C.cyan);
  callout(slide, 8.75, 3.05, 1.75, 1.28, '204', '全量回归', C.paleOrange, C.orange);
  callout(slide, 10.7, 3.05, 1.75, 1.28, '14/14', '训练质量', C.paleBlue, C.blue);
  slide.addShape(S.roundRect, { x: 8.75, y: 4.72, w: 3.7, h: 1.0, rectRadius: 0.06, fill: { color: C.white }, line: { color: C.grid } });
  addText(slide, '交付边界', { x: 8.98, y: 4.93, w: 1.1, h: 0.2, fontSize: 12, bold: true, color: C.navy });
  addText(slide, '单机回环模拟；所有 Worker 共享一张 RTX 4060 Laptop GPU。', { x: 8.98, y: 5.22, w: 3.15, h: 0.3, fontSize: 11.5, color: C.muted, breakLine: true });
  addNotes(slide, '来源：深度学习框架-16 综合项目III.pdf 第 7-9 页；深度学习框架-18 分布式训练-概念与参数服务器架构.pdf 第 2-4 页；深度学习框架-18-2 分布式训练II-Ring AllReduce与评测.pdf 第 2-4 页。');
}

// 3. Architecture
{
  const slide = pptx.addSlide();
  addBase(slide, '系统架构：PS 与 Ring 的职责');
  addText(slide, '两种架构共享同一套 Worker 训练入口，差别集中在梯度聚合拓扑。', { x: 0.7, y: 1.42, w: 8.4, h: 0.25, fontSize: 14, color: C.muted });
  addText(slide, 'Parameter Server（中心化）', { x: 0.88, y: 1.9, w: 4.5, h: 0.3, fontSize: 17, bold: true, color: C.blue });
  node(slide, 2.2, 2.95, 2.0, 0.95, 'PS\n保存参数\n聚合梯度', C.paleBlue, C.blue);
  const psWorkers = [[0.85, 2.12, 'Worker 0'], [4.75, 2.12, 'Worker 1'], [0.85, 4.65, 'Worker 2'], [4.75, 4.65, 'Worker 3']];
  for (const [x, y, t] of psWorkers) {
    node(slide, x, y, 1.35, 0.55, t, C.white);
    arrow(slide, x + (x < 2 ? 1.35 : 0), y + 0.28, x < 2 ? 2.2 : 4.2, 3.42, C.blue);
  }
  addText(slide, 'Push 梯度 → PS 同步聚合 → Worker GPU 更新 / Pull 参数', { x: 0.86, y: 5.55, w: 5.7, h: 0.3, fontSize: 11.2, color: C.muted, align: 'center' });

  addText(slide, 'Ring AllReduce（去中心化）', { x: 7.2, y: 1.9, w: 4.5, h: 0.3, fontSize: 17, bold: true, color: C.cyan });
  const ring = [[8.1, 2.5, 'W0'], [10.7, 2.5, 'W1'], [10.7, 4.65, 'W2'], [8.1, 4.65, 'W3']];
  for (const [x, y, t] of ring) {
    slide.addShape(S.ellipse, { x, y, w: 1.2, h: 0.78, fill: { color: C.paleCyan }, line: { color: C.cyan, pt: 1.1 } });
    addText(slide, t, { x, y: y + 0.29, w: 1.2, h: 0.16, fontSize: 14, bold: true, color: C.cyan, align: 'center' });
  }
  arrow(slide, 9.3, 2.88, 10.7, 2.88, C.cyan); arrow(slide, 11.3, 3.28, 11.3, 4.65, C.cyan);
  arrow(slide, 10.7, 5.04, 9.3, 5.04, C.cyan); arrow(slide, 8.1, 4.65, 8.1, 3.28, C.cyan);
  addText(slide, '每个 Worker 同时发送右邻居、接收左邻居\n完成 N−1 次 ScatterReduce + N−1 次 AllGather', { x: 7.65, y: 5.55, w: 4.7, h: 0.55, fontSize: 12, color: C.muted, align: 'center', breakLine: true });
  addNotes(slide, '来源：深度学习框架-18.pdf 第 20-23、32-39 页；深度学习框架-18-2.pdf 第 7-9、23-25 页。实现中的 PS/Ring 对照统一采用 gRPC/protobuf。');
}

// 4. Communication flow
{
  const slide = pptx.addSlide();
  addBase(slide, '通信与梯度同步流程');
  addText(slide, 'PS 通过中心节点汇总梯度，Ring 将同一梯度切块并沿环传递。', { x: 0.7, y: 1.42, w: 8, h: 0.25, fontSize: 14, color: C.muted });
  const flow = [['Split', '按 N 个 Worker 等长切块\n尾部补零'], ['ScatterReduce', '每轮发送一个块\n边收边累加'], ['AllGather', '再次交换块\n每个 Worker 得到完整梯度']];
  flow.forEach(([title, desc], i) => {
    const x = 0.9 + i * 3.25;
    slide.addShape(S.roundRect, { x, y: 2.1, w: 2.55, h: 1.38, rectRadius: 0.06, fill: { color: i === 1 ? C.paleOrange : C.paleCyan }, line: { color: i === 1 ? C.orange : C.cyan, pt: 1 } });
    addText(slide, title, { x: x + 0.12, y: 2.34, w: 2.31, h: 0.24, fontSize: 17, bold: true, color: i === 1 ? C.orange : C.cyan, align: 'center' });
    addText(slide, desc, { x: x + 0.15, y: 2.75, w: 2.25, h: 0.48, fontSize: 11.5, color: C.ink, align: 'center', breakLine: true });
    if (i < 2) arrow(slide, x + 2.58, 2.78, x + 3.17, 2.78, C.blue);
  });
  slide.addShape(S.roundRect, { x: 1.18, y: 4.35, w: 4.7, h: 1.35, rectRadius: 0.05, fill: { color: C.paleBlue }, line: { color: C.blue } });
  addText(slide, 'PS 通信量', { x: 1.42, y: 4.62, w: 1.3, h: 0.22, fontSize: 16, bold: true, color: C.blue });
  addText(slide, '集群有向发送总量：2NM\n中心节点是带宽与等待的集中点', { x: 2.72, y: 4.53, w: 2.75, h: 0.48, fontSize: 13, color: C.ink, breakLine: true });
  slide.addShape(S.roundRect, { x: 7.0, y: 4.35, w: 4.7, h: 1.35, rectRadius: 0.05, fill: { color: C.paleCyan }, line: { color: C.cyan } });
  addText(slide, 'Ring 通信量', { x: 7.25, y: 4.62, w: 1.45, h: 0.22, fontSize: 16, bold: true, color: C.cyan });
  addText(slide, '集群有向发送总量：2(N−1)M\n单 Worker 量不随 N 线性增长', { x: 8.72, y: 4.53, w: 2.65, h: 0.48, fontSize: 13, color: C.ink, breakLine: true });
  addText(slide, '本项目的 PS/Ring 拓扑对照均使用 gRPC + protobuf；Socket + JSON 仅作为 PS 协议对照。', { x: 1.05, y: 6.25, w: 11.15, h: 0.28, fontSize: 12.5, color: C.muted, align: 'center' });
  addNotes(slide, '来源：深度学习框架-18-2.pdf 第 8-9、19-22 页；通信公式按报告中实际 FP32 梯度字节量口径记录。');
}

// 5. GPU implementation and Nsight
{
  const slide = pptx.addSlide();
  addBase(slide, 'GPU 实现边界与 Nsight 证据');
  addText(slide, '数组管理与原生算子职责分开，避免把 CUDA C 调度收益误写成自写 GEMM 胜出。', { x: 0.7, y: 1.42, w: 10, h: 0.25, fontSize: 14, color: C.muted });
  node(slide, 0.85, 2.1, 2.15, 0.78, '训练图 / Worker\nFP32 梯度', C.white);
  arrow(slide, 3.05, 2.49, 3.75, 2.49, C.blue);
  node(slide, 3.8, 2.1, 2.15, 0.78, 'CuPy\n数组、分配、stream', C.paleCyan, C.cyan);
  arrow(slide, 6.0, 2.49, 6.7, 2.49, C.blue);
  node(slide, 6.75, 2.1, 2.45, 0.78, 'cuda_native_cublas\nConv / Pool + cuBLAS', C.paleBlue, C.blue);
  arrow(slide, 9.25, 2.49, 9.95, 2.49, C.blue);
  node(slide, 10.0, 2.1, 2.15, 0.78, 'Nsight\nSystems / Compute', C.paleOrange, C.orange);
  table(slide, [
    ['阶段一证据', 'P0', 'P1', 'P2', '结论'],
    ['NumPy 组合/ms', '0.168', '12.362', '12.434', '小算力场景 CPU 仍可能更快'],
    ['CuPy 组合/ms', '0.986', '0.929', '0.956', 'GPU 封装基线'],
    ['CUDA C / im2col/ms', '1.314', '1.412', '1.475', '自写路径未全面胜过 CuPy'],
    ['native cuBLAS/ms', '0.315', '0.282', '0.258', '原生调度 + cuBLAS 更快'],
    ['dX Systems/us', '—', '608.3 → 84.7', '—', '有效输出窗口优化'],
  ], { x: 0.85, y: 3.48, w: 8.35, h: 2.28, colW: [2.25, 1.1, 1.4, 1.1, 2.5], rowH: 0.38, fontSize: 9.5 });
  slide.addShape(S.roundRect, { x: 9.55, y: 3.58, w: 2.6, h: 2.05, rectRadius: 0.05, fill: { color: C.paleOrange }, line: { color: C.orange } });
  addText(slide, '解释边界', { x: 9.8, y: 3.84, w: 1.2, h: 0.22, fontSize: 15, bold: true, color: C.orange });
  addText(slide, 'cuBLAS 仍负责 GEMM。\n收益来自 im2col、原生 stream 调度和较少的 Python 封装。\n\nCUDA C / im2col 是独立对照路径，不能与 native cuBLAS 混为一谈。', { x: 9.8, y: 4.22, w: 2.08, h: 1.12, fontSize: 10.6, color: C.ink, breakLine: true });
  addText(slide, '原生 Conv/Pool 与 Adam 数学规则保持不变；ResNet 使用 base_width=16，冻结 512 张训练图校准的 BN。', { x: 0.9, y: 6.05, w: 11.4, h: 0.36, fontSize: 12.2, color: C.muted, align: 'center' });
  addNotes(slide, '来源：深度学习框架-16 综合项目III.pdf 第 9 页；项目第一阶段 README.md 第 2、4、5 节；native_cublas_report.md。');
}

// 6. Experimental design
{
  const slide = pptx.addSlide();
  addBase(slide, '实验设计与公平测量口径');
  table(slide, [
    ['任务', '模型 / 数据', '正式配置', '质量指标'],
    ['MNIST', '784→64→10 MLP\n50,000 train / 10,000 val', 'single；PS JSON/gRPC 2/4；Ring gRPC 2/4', '验证 accuracy ≥95%'],
    ['DonkeyCar', 'ResNet18 base_width=16\n8,000 / 1,000 / 1,000', 'single；PS JSON/gRPC 2/4；Ring gRPC 2/4', 'angle MAE；相对初始化下降'],
  ], { x: 0.75, y: 1.65, w: 11.85, h: 2.1, colW: [1.2, 3.3, 4.45, 2.9], rowH: 0.68, fontSize: 11.5 });
  const bullets = [
    ['同一初始点', 'seed=0、FP32、统一数据顺序和 checkpoint'],
    ['同一计时口径', '完整第 1 个 epoch，3 次独立性能重复，取中位数'],
    ['同一资源边界', '所有 Worker 共享一张 RTX 4060 Laptop GPU'],
    ['BN 控制', '仅用 512 张训练图校准 running mean/variance，随后冻结'],
  ];
  bullets.forEach(([label, desc], i) => {
    const y = 4.25 + i * 0.55;
    slide.addShape(S.ellipse, { x: 0.95, y: y + 0.05, w: 0.22, h: 0.22, fill: { color: i % 2 ? C.cyan : C.blue }, line: { color: i % 2 ? C.cyan : C.blue } });
    addText(slide, label, { x: 1.35, y, w: 1.45, h: 0.2, fontSize: 13, bold: true, color: C.navy });
    addText(slide, desc, { x: 2.85, y, w: 8.8, h: 0.2, fontSize: 12.5, color: C.ink });
  });
  slide.addShape(S.roundRect, { x: 10.25, y: 4.35, w: 2.25, h: 1.7, rectRadius: 0.05, fill: { color: C.paleOrange }, line: { color: C.orange } });
  addText(slide, '注意', { x: 10.5, y: 4.62, w: 0.8, h: 0.22, fontSize: 15, bold: true, color: C.orange });
  addText(slide, '这是单机回环模拟。\n通信结果不能直接外推到多机网络。', { x: 10.5, y: 5.02, w: 1.7, h: 0.55, fontSize: 11.5, color: C.ink, breakLine: true });
  addNotes(slide, '来源：深度学习框架-18.pdf 第 2-4、29-30、64-67 页；优化版报告“固定配置与测量口径”和“ResNet18 配置与 BN 控制”章节。');
}

// 7. PS results
{
  const slide = pptx.addSlide();
  addBase(slide, 'PS 结果：gRPC 大幅降低通信开销');
  const psRows = [
    ['任务 / 协议', 'Worker', 'epoch/s', '计算/s', '通信/s', '正文 MB'],
    ['MNIST JSON', '2', '33.217', '0.993', '28.490', mb(num(get('mnist_mlp', 'ps-json-2'), 'request_bytes') + num(get('mnist_mlp', 'ps-json-2'), 'response_bytes'))],
    ['MNIST gRPC', '2', '12.207', '0.810', '5.850', mb(num(get('mnist_mlp', 'ps-grpc-2'), 'request_bytes') + num(get('mnist_mlp', 'ps-grpc-2'), 'response_bytes'))],
    ['MNIST JSON', '4', '52.964', '1.873', '42.595', mb(num(get('mnist_mlp', 'ps-json-4'), 'request_bytes') + num(get('mnist_mlp', 'ps-json-4'), 'response_bytes'))],
    ['MNIST gRPC', '4', '15.399', '0.962', '6.885', mb(num(get('mnist_mlp', 'ps-grpc-4'), 'request_bytes') + num(get('mnist_mlp', 'ps-grpc-4'), 'response_bytes'))],
    ['ResNet JSON', '2', '273.579', '18.125', '244.177', mb(num(get('donkey_resnet18', 'ps-json-2'), 'request_bytes') + num(get('donkey_resnet18', 'ps-json-2'), 'response_bytes'))],
    ['ResNet gRPC', '2', '14.073', '6.257', '3.491', mb(num(get('donkey_resnet18', 'ps-grpc-2'), 'request_bytes') + num(get('donkey_resnet18', 'ps-grpc-2'), 'response_bytes'))],
    ['ResNet JSON', '4', '462.667', '18.905', '416.378', mb(num(get('donkey_resnet18', 'ps-json-4'), 'request_bytes') + num(get('donkey_resnet18', 'ps-json-4'), 'response_bytes'))],
    ['ResNet gRPC', '4', '16.936', '6.312', '5.068', mb(num(get('donkey_resnet18', 'ps-grpc-4'), 'request_bytes') + num(get('donkey_resnet18', 'ps-grpc-4'), 'response_bytes'))],
  ];
  table(slide, psRows, { x: 0.72, y: 1.5, w: 8.55, h: 4.55, colW: [1.7, 0.8, 1.25, 1.25, 1.25, 1.45], rowH: 0.47, fontSize: 10.3 });
  slide.addShape(S.roundRect, { x: 9.65, y: 1.68, w: 2.55, h: 1.2, rectRadius: 0.05, fill: { color: C.paleCyan }, line: { color: C.cyan } });
  addText(slide, 'MNIST gRPC / JSON', { x: 9.9, y: 1.93, w: 2.05, h: 0.2, fontSize: 13, bold: true, color: C.cyan, align: 'center' });
  addText(slide, '2 Worker：2.72×\n4 Worker：3.44×', { x: 9.95, y: 2.24, w: 1.95, h: 0.4, fontSize: 14, bold: true, color: C.navy, align: 'center', breakLine: true });
  slide.addShape(S.roundRect, { x: 9.65, y: 3.18, w: 2.55, h: 1.2, rectRadius: 0.05, fill: { color: C.paleBlue }, line: { color: C.blue } });
  addText(slide, 'ResNet gRPC / JSON', { x: 9.9, y: 3.43, w: 2.05, h: 0.2, fontSize: 13, bold: true, color: C.blue, align: 'center' });
  addText(slide, '2 Worker：19.44×\n4 Worker：27.33×', { x: 9.95, y: 3.74, w: 1.95, h: 0.4, fontSize: 14, bold: true, color: C.navy, align: 'center', breakLine: true });
  addText(slide, 'PS 的中心聚合逻辑保留；优化主要减少 protobuf 数组路径中的拷贝、序列化和确认等待。', { x: 9.7, y: 4.86, w: 2.45, h: 0.65, fontSize: 11.2, color: C.muted, align: 'center', breakLine: true });
  addNotes(slide, '来源：深度学习框架-18.pdf 第 39、64-67 页；优化版 comparison.csv。正文 MB 为 request_bytes + response_bytes，未计 HTTP/2、TCP/IP 头。');
}

// 8. Ring results
{
  const slide = pptx.addSlide();
  addBase(slide, 'Ring AllReduce 结果：通信更均匀，单卡仍限制扩展');
  const ringRows = [
    ['任务', 'Worker', 'epoch/s', '计算/s', '通信/s', '加速比'],
    ['MNIST', '2', '3.369', '0.725', '0.820', '0.411'],
    ['MNIST', '4', '5.684', '1.069', '2.184', '0.244'],
    ['ResNet18', '2', '13.187', '7.592', '1.558', '0.686'],
    ['ResNet18', '4', '15.422', '7.726', '2.066', '0.587'],
  ];
  table(slide, ringRows, { x: 0.75, y: 1.55, w: 6.0, h: 2.45, colW: [1.25, 0.72, 0.98, 1.0, 1.0, 1.0], rowH: 0.48, fontSize: 10.5 });
  slide.addChart(pptx.ChartType.line, [
    { name: '理想线性', labels: ['1', '2', '4'], values: [1, 2, 4] },
    { name: 'MNIST Ring', labels: ['1', '2', '4'], values: [1, 0.411, 0.244] },
    { name: 'ResNet18 Ring', labels: ['1', '2', '4'], values: [1, 0.686, 0.587] },
  ], {
    x: 7.05, y: 1.46, w: 5.55, h: 3.15, chartColors: ['9AA5B1', C.cyan, C.blue],
    showLegend: true, legendPos: 'b', fontFace: FONT, catAxisLabelFontFace: FONT, valAxisLabelFontFace: FONT,
    legendFontFace: FONT, dataLabelFontFace: FONT,
    catAxisLabelFontSize: 10, valAxisLabelFontSize: 10, valAxisMinVal: 0, valAxisMaxVal: 4.2, valAxisMajorUnit: 1,
    valGridLine: { color: C.grid, size: 0.7 }, catGridLine: { style: 'none' }, lineSize: 2.2, lineDataSymbol: 'circle',
    chartArea: { fill: { color: C.white }, border: { color: C.grid, pt: 0.8 } },
    showTitle: true, title: '实测加速比与理想线性参考', titleFontFace: FONT, titleFontSize: 14,
  });
  slide.addShape(S.roundRect, { x: 0.9, y: 4.55, w: 5.65, h: 1.25, rectRadius: 0.05, fill: { color: C.paleCyan }, line: { color: C.cyan } });
  addText(slide, 'Ring 的工程优势', { x: 1.18, y: 4.82, w: 1.55, h: 0.24, fontSize: 12.5, bold: true, color: C.cyan });
  addText(slide, '没有 PS 中心带宽热点；每个 rank 只和相邻节点交换分块，满足 Split → ScatterReduce → AllGather。', { x: 2.82, y: 4.68, w: 3.25, h: 0.48, fontSize: 11.4, color: C.ink, breakLine: true });
  slide.addShape(S.roundRect, { x: 7.05, y: 4.95, w: 5.55, h: 0.85, rectRadius: 0.05, fill: { color: C.paleOrange }, line: { color: C.orange } });
  addText(slide, '瓶颈判断：多进程共享同一 GPU，通信更均匀但计算资源没有增加；进程调度与同步开销仍高于单进程。', { x: 7.32, y: 5.2, w: 5.0, h: 0.3, fontSize: 11.6, color: C.ink, align: 'center' });
  addNotes(slide, '来源：深度学习框架-18-2.pdf 第 4、8-9、19-22、39-41 页；优化版 comparison.csv 与 ring_chunk_flow.json。');
}

// 9. Baseline vs optimized
{
  const slide = pptx.addSlide();
  addBase(slide, '优化前后对照：优化版作为主结果');
  const configs = [
    ['MNIST PS2', get('mnist_mlp', 'ps-grpc-2')], ['MNIST PS4', get('mnist_mlp', 'ps-grpc-4')],
    ['MNIST R2', get('mnist_mlp', 'ring-grpc-2')], ['MNIST R4', get('mnist_mlp', 'ring-grpc-4')],
    ['ResNet PS2', get('donkey_resnet18', 'ps-grpc-2')], ['ResNet PS4', get('donkey_resnet18', 'ps-grpc-4')],
    ['ResNet R2', get('donkey_resnet18', 'ring-grpc-2')], ['ResNet R4', get('donkey_resnet18', 'ring-grpc-4')],
  ];
  const baseTimes = [34.329, 62.145, 4.226, 6.396, 110.171, 210.157, 13.276, 16.488];
  const optTimes = configs.map(([, row]) => num(row, 'epoch_s_median'));
  slide.addChart(pptx.ChartType.bar, [
    { name: '原基线', labels: configs.map(([label]) => label), values: baseTimes },
    { name: '优化版', labels: configs.map(([label]) => label), values: optTimes },
  ], {
    x: 0.7, y: 1.5, w: 8.0, h: 4.55, barDir: 'col', catAxisLabelFontFace: FONT, valAxisLabelFontFace: FONT,
    legendFontFace: FONT, dataLabelFontFace: FONT,
    catAxisLabelFontSize: 9, valAxisLabelFontSize: 10, chartColors: [C.orange, C.cyan], showLegend: true, legendPos: 'b',
    showValue: false, valAxisMinVal: 0, valGridLine: { color: C.grid, size: 0.7 }, catGridLine: { style: 'none' },
    chartArea: { fill: { color: C.white }, border: { color: C.grid, pt: 0.8 } },
    showTitle: true, title: '同配置 epoch 中位耗时（秒）', titleFontFace: FONT, titleFontSize: 14,
  });
  callout(slide, 9.15, 1.62, 1.55, 1.15, '42.36%', 'gRPC 分布式整体中位下降', C.paleCyan, C.cyan);
  callout(slide, 10.95, 1.62, 1.55, 1.15, '81.22%', 'PS gRPC 中位下降', C.paleBlue, C.blue);
  callout(slide, 9.15, 3.05, 1.55, 1.15, '8.79%', 'Ring 中位下降', C.paleOrange, C.orange);
  callout(slide, 10.95, 3.05, 1.55, 1.15, '< 1', '分布式相对单进程加速比', C.paleBlue, C.blue);
  addText(slide, '判定：8 个 gRPC/protobuf 分布式配置均比原基线快，质量 14/14、全量回归 204 项通过，因此优化版作为主结果；原基线保留作历史参照。', { x: 9.05, y: 4.75, w: 3.45, h: 0.75, fontSize: 11.5, color: C.ink, align: 'center', breakLine: true });
  addNotes(slide, '来源：优化版 optimization_comparison.csv / delivery_summary.json。下降率按 (基线 epoch - 优化版 epoch) / 基线 epoch 计算。');
}

// 10. Acceptance and conclusion
{
  const slide = pptx.addSlide();
  addBase(slide, '验收结论与需要继续说明的限制');
  const checks = [
    ['PS：MNIST + DonkeyCar', '已完成', C.cyan],
    ['Ring：Split / ScatterReduce / AllGather', '已完成', C.cyan],
    ['计算 / 通信 / 消息量 / 质量 / 加速比', '已记录', C.cyan],
    ['Nsight Systems / Compute', '已留档', C.cyan],
    ['ResNet 独立整批逐步梯度等价', '未通过原门槛', C.orange],
  ];
  checks.forEach(([label, status, color], i) => {
    const y = 1.55 + i * 0.72;
    slide.addShape(S.roundRect, { x: 0.85, y, w: 5.35, h: 0.5, rectRadius: 0.04, fill: { color: i === 4 ? C.paleOrange : C.white }, line: { color: i === 4 ? C.orange : C.grid } });
    addText(slide, label, { x: 1.08, y: y + 0.15, w: 3.65, h: 0.16, fontSize: 12.5, color: C.ink });
    addText(slide, status, { x: 4.8, y: y + 0.13, w: 1.15, h: 0.24, fontSize: 11.5, bold: true, color, align: 'right' });
  });
  slide.addShape(S.roundRect, { x: 6.85, y: 1.55, w: 5.6, h: 3.25, rectRadius: 0.06, fill: { color: C.navy }, line: { color: C.navy } });
  addText(slide, '汇报结论', { x: 7.2, y: 1.95, w: 1.4, h: 0.25, fontSize: 18, bold: true, color: '8DE0D9' });
  addText(slide, '在单机单卡模拟环境中，PS 与 Ring 的实现、记录和对照满足第四周汇报需要。优化降低了通信与封装开销，但分布式加速比仍低于 1。', { x: 7.2, y: 2.45, w: 4.8, h: 0.75, fontSize: 15, color: C.white, breakLine: true });
  addText(slide, '解释结果时应强调：共享 GPU 是性能边界；ResNet 的原独立整批逐步梯度门槛失败需要单独披露，不能与归约或优化器更新检查混称。', { x: 7.2, y: 3.58, w: 4.8, h: 0.7, fontSize: 12.5, color: 'D9E2EC', breakLine: true });
  addText(slide, '材料：优化版 README、optimization_comparison.csv、delivery_summary.json、204 项回归日志。', { x: 1.0, y: 6.45, w: 11.3, h: 0.24, fontSize: 12, color: C.muted, align: 'center' });
  addNotes(slide, '结论依据：优化版 delivery_summary.json、README.md 的验收边界与限制章节。ResNet 逐步梯度等价失败保留为独立限制，归约与 CPU/GPU 更新检查分别通过。');
}

const archive = await JSZip.loadAsync(await pptx.write({ outputType: 'nodebuffer' }));
for (const part of Object.values(archive.files)) {
  if (!part.name.endsWith('.xml')) continue;
  let xml = await part.async('string');
  if (/<a:ext\b[^>]*\bc[xy]="-/.test(xml)) {
    throw new Error(`Invalid negative DrawingML extent in ${part.name}`);
  }
  if (/^ppt\/slides\/slide\d+\.xml$/.test(part.name)) {
    // PptxGenJS tables can reuse another shape's ID. This deck has no
    // ID-linked connectors/animations, so assign unique IDs in drawing order.
    if (/<a:(?:stCxn|endCxn)\b|<p:timing\b/.test(xml)) {
      throw new Error(`Shape-ID references need explicit remapping in ${part.name}`);
    }
    let shapeId = 0;
    xml = xml.replace(/(<p:cNvPr\b[^>]*\bid=")[^"]+("[^>]*>)/g,
      (_, before, after) => `${before}${++shapeId}${after}`);
    archive.file(part.name, xml);
  }
  if (part.name === '[Content_Types].xml') {
    // The generator declares one slide master per slide but emits one shared
    // master. Drop only unused declarations; referenced parts are preserved.
    xml = xml.replace(/<Override\s+PartName="([^"]+)"[^>]*\/>/g,
      (entry, target) => archive.file(target.replace(/^\//, '')) ? entry : '');
    archive.file(part.name, xml);
  }
  if (part.name === 'ppt/presentation.xml') {
    // Notes masters precede the slide list in the PresentationML sequence.
    xml = xml.replace(/(<p:sldIdLst>[\s\S]*?<\/p:sldIdLst>)(<p:notesMasterIdLst>[\s\S]*?<\/p:notesMasterIdLst>)/, '$2$1');
    archive.file(part.name, xml);
  }
  if (part.name.startsWith('ppt/charts/')) {
    // PptxGenJS 4.0.1 writes bar-only flags into line series and a third axis
    // for 2D charts. Keep chart data/workbooks intact and normalize these tags.
    xml = xml.replace(/<c:(line|bar)Chart>[\s\S]*?<\/c:\1Chart>/g, (chart, kind) => {
      if (kind === 'line') {
        if (!chart.includes('<c:grouping ')) {
          chart = chart.replace('<c:lineChart>', '<c:lineChart><c:grouping val="standard"/>');
        }
        chart = chart.replace(/<c:invertIfNegative\b[^>]*\/>/g, '');
        chart = chart.replace(/(<c:dLbls>[\s\S]*?<\/c:dLbls>)(<c:marker>[\s\S]*?<\/c:marker>)/g, '$2$1');
      }
      let axisCount = 0;
      chart = chart.replace(/<c:axId\s+val="[^"]+"\s*\/>/g, axis => ++axisCount <= 2 ? axis : '');
      if (axisCount < 2) throw new Error(`Missing 2D chart axes in ${part.name}`);
      return chart;
    });
    // Specify an East Asian font for Chinese chart titles and labels as well.
    xml = xml.replace(/(<a:latin\b[^>]*\/>)\s*(?!<a:ea\b)/g, `$1<a:ea typeface="${FONT}"/>`);
    // These charts have a single category level. Use the standard flat cache
    // without changing any workbook reference, category label or value.
    xml = xml.replace(/<c:multiLvlStrRef>[\s\S]*?<\/c:multiLvlStrRef>/g, ref => {
      if ((ref.match(/<c:lvl>/g) ?? []).length !== 1) return ref;
      return ref.replaceAll('multiLvlStrRef', 'strRef').replaceAll('multiLvlStrCache', 'strCache')
        .replace(/<\/?c:lvl>/g, '');
    });
    archive.file(part.name, xml);
  }
}
// Re-export through the presentation runtime. The raw PptxGenJS package can
// pass XML validation but still be rejected by Microsoft PowerPoint. Keep
// intermediate files and the runtime's inspection sidecar outside the report.
const { FileBlob, PresentationFile } = await import(
  pathToFileURL(require.resolve('@oai/artifact-tool')).href
);
const buildRoot = path.join(repo, '.codex');
fs.mkdirSync(buildRoot, { recursive: true });
const buildDir = fs.mkdtempSync(path.join(buildRoot, 'week4-pptx-'));
const intermediatePath = path.join(buildDir, 'normalized.pptx');
const compatiblePath = path.join(buildDir, 'compatible.pptx');
fs.writeFileSync(intermediatePath, await archive.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' }));
const presentation = await PresentationFile.importPptx(await FileBlob.load(intermediatePath));
await (await PresentationFile.exportPptx(presentation)).save(compatiblePath);
// The runtime preserves chart formulas/caches but omits embedded workbooks.
// Restore the original sheets so PowerPoint's Edit Data remains available.
const compatible = await JSZip.loadAsync(fs.readFileSync(compatiblePath));
const chartParts = Object.values(compatible.files).filter(part =>
  /\/charts\/chart\d+\.xml$/.test(part.name));
if (chartParts.length !== 2) throw new Error('Expected both report charts');
for (const part of chartParts) {
  const sourcePath = `ppt/charts/${path.posix.basename(part.name)}`;
  const sourceXml = await archive.file(sourcePath).async('string');
  let chartXml = await part.async('string');
  const formulas = xml => [...xml.matchAll(/<c:f>([\s\S]*?)<\/c:f>/g)].map(match => match[1]);
  if (JSON.stringify(formulas(sourceXml)) !== JSON.stringify(formulas(chartXml))) {
    throw new Error(`Chart formula mapping changed: ${part.name}`);
  }
  const sourceSeries = [...sourceXml.matchAll(/<c:ser>[\s\S]*?<\/c:ser>/g)].map(match => match[0]);
  let seriesIndex = 0;
  chartXml = chartXml.replace(/<c:ser>[\s\S]*?<\/c:ser>/g, series => {
    const originalSeries = sourceSeries[seriesIndex++];
    for (const tag of ['tx', 'cat', 'val']) {
      const pattern = new RegExp(`<c:${tag}>[\\s\\S]*?<\\/c:${tag}>`);
      const originalData = originalSeries?.match(pattern)?.[0];
      if (!originalData || !pattern.test(series)) throw new Error(`Missing chart data: ${part.name}/${tag}`);
      series = series.replace(pattern, () => originalData);
    }
    return series;
  });
  if (seriesIndex !== sourceSeries.length) throw new Error(`Chart series count changed: ${part.name}`);
  const sourceRels = `ppt/charts/_rels/${path.posix.basename(part.name)}.rels`;
  let rels = await archive.file(sourceRels).async('string');
  const targets = [...rels.matchAll(/Target="([^"]+)"/g)];
  if (targets.length !== 1 || !targets[0][1].endsWith('.xlsx')) {
    throw new Error(`Unexpected chart workbook relationship: ${sourceRels}`);
  }
  const sourceTarget = targets[0][1];
  const workbookPath = path.posix.normalize(path.posix.join(path.posix.dirname(sourcePath), sourceTarget));
  compatible.file(workbookPath, await archive.file(workbookPath).async('nodebuffer'));
  rels = rels.replace(`Target="${sourceTarget}"`,
    `Target="${path.posix.relative(path.posix.dirname(part.name), workbookPath)}"`);
  compatible.file(`${path.posix.dirname(part.name)}/_rels/${path.posix.basename(part.name)}.rels`, rels);
  const externalData = sourceXml.match(/<c:externalData\b[\s\S]*?<\/c:externalData>/)?.[0];
  if (!externalData || /<c:externalData\b/.test(chartXml)) {
    throw new Error(`Unexpected embedded-data state: ${part.name}`);
  }
  chartXml = chartXml.replace('</c:chartSpace>', externalData.replace('<c:externalData ',
    '<c:externalData xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" ') + '</c:chartSpace>');
  compatible.file(part.name, chartXml);
}
let contentTypes = await compatible.file('[Content_Types].xml').async('string');
if (!/Extension="xlsx"/.test(contentTypes)) {
  contentTypes = contentTypes.replace('</Types>',
    '<Default Extension="xlsx" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"/></Types>');
  compatible.file('[Content_Types].xml', contentTypes);
}
fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, await compatible.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' }));
console.log(outPath);
