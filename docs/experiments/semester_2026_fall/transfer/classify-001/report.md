# 分类迁移的六档递进实验

每档使用两个种子，输入 224×224，batch 16，训练 2 轮。道路数据按时间段划分，转向分档边界只由训练标签确定。CIFAR-10 的三个平衡子集全部来自官方训练档案，规模为 3,000/1,000/1,000。折线表示各轮训练与验证准确率的种子均值。

![训练与验证曲线](curves.png)

| 数据 | 方法 | 训练准确率 | 验证准确率 | 训练减验证 |
| --- | --- | --- | --- | --- |
| donkey | small | 0.83005 ± 0.02074 | 0.93625 ± 0.02298 | -0.10620 ± 0.00224 |
| donkey | dropout | 0.79928 ± 0.02414 | 0.92250 ± 0.02121 | -0.12322 ± 0.00292 |
| donkey | augment-low-lr | 0.47476 ± 0.06833 | 0.51000 ± 0.00707 | -0.03524 ± 0.07540 |
| donkey | cached-extractor | 0.78822 ± 0.00578 | 0.87500 ± 0.00000 | -0.08678 ± 0.00578 |
| donkey | augmented-extractor | 0.74375 ± 0.00136 | 0.85875 ± 0.01945 | -0.11500 ± 0.01809 |
| donkey | fine-tune-last-two | 0.73534 ± 0.01598 | 0.84500 ± 0.03536 | -0.10966 ± 0.01938 |
| cifar10 | small | 0.33422 ± 0.02600 | 0.31650 ± 0.00495 | 0.01772 ± 0.03095 |
| cifar10 | dropout | 0.27807 ± 0.02080 | 0.28850 ± 0.01344 | -0.01043 ± 0.00736 |
| cifar10 | augment-low-lr | 0.18382 ± 0.00095 | 0.22850 ± 0.00919 | -0.04468 ± 0.01014 |
| cifar10 | cached-extractor | 0.79579 ± 0.00047 | 0.77700 ± 0.00283 | 0.01879 ± 0.00330 |
| cifar10 | augmented-extractor | 0.71591 ± 0.00851 | 0.75500 ± 0.00141 | -0.03909 ± 0.00709 |
| cifar10 | fine-tune-last-two | 0.51337 ± 0.01134 | 0.71000 ± 0.01131 | -0.19663 ± 0.02266 |

small 从零训练，dropout 加入 0.3 随机失活，augment-low-lr 同时增加图像增强并将 lr 降至 0.0001。后面三档使用 ImageNet 权重。cached-extractor 只训练任务头，augmented-extractor 每轮重新增强图像。fine-tune-last-two 在第二轮释放末两段，任务头 lr=0.0001、骨干倍率 0.1。

summary.json 和每个 attempt 的 config、metrics、result 与 checkpoint 位于本目录。复现命令为 `python -m benchmark.transfer_study classification --out runs/stage3/classify-repro`。
