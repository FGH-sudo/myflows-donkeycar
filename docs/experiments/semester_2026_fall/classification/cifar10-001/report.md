# CIFAR-10 图像分类系统测试

数据为官方二进制包。训练档案内按类固定划分 45,000 / 5,000。配置采用 16 通道的小 CNN、batch 64、Adam lr=0.001、种子 1，共 3 轮。MyFlows 与 PyTorch 读取同一初值 NPZ 与逐轮 batch 索引，每轮使用 44,992 个训练样本。两种框架均评估全部 5,000 个验证样本。

| epoch | MyFlows 训练准确率 | MyFlows 验证准确率 | PyTorch 验证准确率 |
| --- | --- | --- | --- |
| 1 | 41.87% | 50.24% | 50.54% |
| 2 | 53.76% | 56.46% | 56.36% |
| 3 | 57.95% | 60.22% | 60.22% |

末轮验证准确率相差 +0.000 个百分点（MyFlows 减 PyTorch），满足 3 个百分点门槛。按验证准确率锁定 checkpoint 后，官方 10,000 张测试图像一次评估的准确率为 60.13%，交叉熵 1.1558，多数类基线 10%。

日常自动检查使用 32 张固定平衡的真实训练图像，达到 95% 过拟合门槛。ResNet18 cifar stem 与小 CNN 均完成独立短训练。验证证据见 `runs/stage3/cache-cifar-check.log`、`runs/stage3/cifar-resnet-smoke/` 和 `runs/stage3/cifar-smoke/`。

本目录的 split-and-order.npz、initial.npz、metrics.jsonl、resources.jsonl、pytorch-validation.json、locked.json 和 official-test.json 保存原始协议与结果。锁定 checkpoint SHA-256 为 `ed22f8d9a7f008af964add73bf185b18117c6af46c146588fb5dcd9f27aebef7`。

复现时选用新运行目录，依次运行：

```powershell
python -m apps.train.train_classify --run-dir runs/stage3/cifar-repro --epochs 3 --batch 64 --width 16 --seed 1
python -m benchmark.cifar10_reference --run runs/stage3/cifar-repro
python -m apps.train.train_classify --score-test --run-dir runs/stage3/cifar-repro
```
