# ops-001

状态：通过

## 算子输出相对 FP64 的最大误差（max|y − y64| / max|y64|）

conv：4×16×20×24 输入、32×16×3×3 卷积核、stride 1、padding 1；bn：8×16×12×12，unit 为 N(0,1)，offset 为 1000 + 0.01·N(0,1)（均值是标准差的 1e5 倍）；ce：64×10 logits，unit 为 N(0,1)，large 为 1e4 倍，one-hot 标签；adam：1000 个参数、10 步、梯度量级 1e-6–10，参考为教科书 Adam（零初始化矩估计 + 偏差修正）的 FP64 结果，修改前一栏反映的是旧实现的语义偏差而不只是舍入误差。

| 输出 | 修改前 | 修改后 | 倍数 |
|---|---|---|---|
| adam/cpu/step1 | 2.22e-04 | 3.86e-08 | 5760.4× |
| adam/cpu/step10 | 5.70e-04 | 1.65e-07 | 3451.6× |
| adam/cuda/step1 | 2.22e-04 | 3.86e-08 | 5760.4× |
| adam/cuda/step10 | 5.70e-04 | 1.65e-07 | 3451.6× |
| bn/cpu/offset/dbeta | 8.89e-08 | 1.34e-08 | 6.6× |
| bn/cpu/offset/dgamma | 8.29e-03 | 3.88e-08 | 213921.2× |
| bn/cpu/offset/dx | 1.69e-04 | 9.30e-08 | 1820.1× |
| bn/cpu/offset/y | 1.93e-03 | 6.82e-08 | 28336.5× |
| bn/cpu/unit/dbeta | 8.89e-08 | 1.34e-08 | 6.6× |
| bn/cpu/unit/dgamma | 1.29e-07 | 3.02e-08 | 4.3× |
| bn/cpu/unit/dx | 1.87e-07 | 1.21e-07 | 1.5× |
| bn/cpu/unit/y | 1.24e-07 | 9.81e-08 | 1.3× |
| bn/cuda/offset/dbeta | 7.79e-08 | 1.34e-08 | 5.8× |
| bn/cuda/offset/dgamma | 3.06e-03 | 3.88e-08 | 78962.9× |
| bn/cuda/offset/dx | 1.61e-04 | 9.30e-08 | 1730.7× |
| bn/cuda/offset/y | 1.93e-03 | 8.77e-08 | 22037.9× |
| bn/cuda/unit/dbeta | 7.79e-08 | 1.34e-08 | 5.8× |
| bn/cuda/unit/dgamma | 9.00e-08 | 3.02e-08 | 3.0× |
| bn/cuda/unit/dx | 1.87e-07 | 1.21e-07 | 1.5× |
| bn/cuda/unit/y | 1.24e-07 | 9.11e-08 | 1.4× |
| ce/cpu/large/grad | 7.53e-112 | 7.53e-112 | 1.0× |
| ce/cpu/large/loss | 9.99e-01 | 3.59e-08 | 27856345.1× |
| ce/cpu/unit/grad | 4.72e-08 | 3.53e-08 | 1.3× |
| ce/cpu/unit/loss | 6.43e-08 | 1.30e-08 | 4.9× |
| ce/cuda/large/grad | 7.53e-112 | 7.53e-112 | 1.0× |
| ce/cuda/large/loss | 9.99e-01 | 3.59e-08 | 27856345.1× |
| ce/cuda/unit/grad | 4.25e-08 | 3.53e-08 | 1.2× |
| ce/cuda/unit/loss | 1.30e-08 | 1.30e-08 | 1.0× |
| conv/cuda_native_cublas/db | 6.06e-07 | 3.65e-08 | 16.6× |
| conv/cuda_native_cublas/dw | 1.37e-07 | 1.37e-07 | 1.0× |
| conv/cuda_native_cublas/dx | 1.83e-07 | 1.83e-07 | 1.0× |
| conv/cuda_native_cublas/y | 1.42e-07 | 1.42e-07 | 1.0× |
| conv/cupy/db | 1.02e-07 | 3.65e-08 | 2.8× |
| conv/cupy/dw | 1.37e-07 | 1.37e-07 | 1.0× |
| conv/cupy/dx | 1.41e-07 | 1.50e-07 | 0.9× |
| conv/cupy/y | 1.42e-07 | 1.42e-07 | 1.0× |
| conv/numpy/db | 1.74e-07 | 3.65e-08 | 4.8× |
| conv/numpy/dw | 3.24e-07 | 3.24e-07 | 1.0× |
| conv/numpy/dx | 1.50e-07 | 1.50e-07 | 1.0× |
| conv/numpy/y | 3.57e-07 | 3.57e-07 | 1.0× |

原始证据：config.json、results.json、timings.csv。
