# 开发环境

本项目的原生 CUDA 路径和 Python 实验路径分开管理：系统工具链负责原生库构建（经 CMake，Windows 用 MSVC，Linux 用 GCC/Clang）以及 `nvcc` 和 Nsight；隔离的 `.venv` 负责 CuPy、PyTorch 与课程项目依赖。CuPy 需要的 CUDA 12.6 运行库和 NVRTC 头文件也固定安装在 `.venv` 内，避免被系统 Toolkit 13.3 的头文件覆盖。下文命令以本机 Windows（PowerShell）为准，Linux 的写法和差异见 [Linux 环境](#linux-环境)。

## 已安装工具（本机 Windows）

- NVIDIA CUDA Toolkit 13.3.1，`nvcc` 版本 13.3.73。
- Visual Studio Community 2022 17.14.27，MSVC 14.44.35207，Windows SDK 10.0.26100.0。
- CMake 4.4.3。
- Nsight Compute 2026.2.1，命令行程序位于 `C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\target\windows-desktop-win7-x64\ncu.exe`。
- `uv` 0.11.9。

## Python 依赖

根目录的 `pyproject.toml` 和 `uv.lock` 是项目 Python 依赖的声明与锁定文件。当前使用 `cupy-cuda12x==14.0.1`，对应 CUDA 12 系列；`.venv` 内固定安装 `nvidia-cublas-cu12`、`nvidia-cuda-nvrtc-cu12` 和 `nvidia-cuda-runtime-cu12`。PyTorch wheel 自带同名的 `nvrtc64_120_0.dll`（12.8）：进程里先加载哪一个，CuPy 就用哪一个，而 12.8 的 NVRTC 配 12.6 的头文件会让 CUB kernel（如布尔规约）编译失败。因此 `MyFlows/core/device.py` 只用 `importlib.util.find_spec` 定位 `torch/lib`（供 ONNX Runtime 加载 cuDNN），不再导入 torch；原生库也优先加载 `.venv` 内的 NVRTC 12.6 与 cuBLAS。实际版本可由 `MyFlows.ops.cuda_native.info()` 读出（`nvrtc`、`binary`、`device_arch` 等字段），实验应记录该输出而不是只看包版本。

同步依赖时使用：

```powershell
uv sync --locked --inexact
uv pip check --python .venv\Scripts\python.exe
```

默认 `dev` 依赖组包括统一测试入口需要的 FastAPI、httpx 和 python-multipart，并限定 `setuptools<82`，与当前 PyTorch wheel 的依赖约束一致。默认同时安装的 `console` 依赖组供训练控制台使用：FastAPI、uvicorn 和提供 `pynvml` 的 `nvidia-ml-py`（没有 NVML 时 GPU 采样退回 nvidia-smi）。默认还安装 `onnx` 组（onnxruntime-gpu，供 ONNX 评估、FastAPI/gRPC 推理服务、INT8 量化和驾驶 pilot 使用，CUDA/cuDNN DLL 复用 PyTorch wheel 自带的版本）和 `donkeycar` 组（donkeycar 5.3.0、gymnasium 1.3.0 和固定到 GitHub 提交的 gym-donkeycar，供 `mycar/manage.py` 模拟器驾驶）。donkeycar 5.3.0 声明 `numpy<2`，`pyproject.toml` 用 `override-dependencies` 统一为 numpy 2.2.6，因此 `uv pip check` 会报告这一条不兼容，属预期，其余应无报错。`--inexact` 保留本地 wheel 安装的 PyTorch 及其额外依赖；普通的精确 `uv sync` 会移除未声明在锁文件中的包。

## 训练控制台前端

`apps/console/web` 是独立的 Node 工程，依赖由 `package-lock.json` 锁定，`node_modules/` 与构建产物 `dist/` 不纳入 Git。Vite 8 要求 Node.js 20.19+ 或 22.12+（当前机器为 Node 24.16）。

```powershell
Set-Location apps\console\web
npm ci          # 按锁文件安装
npm run build   # 类型检查并输出 dist/，由 python -m apps.console.server 托管
npm run dev     # 开发模式，http://127.0.0.1:5173，/api 代理到 8790（可用 CONSOLE_BACKEND 改）
npm run lint    # oxlint
```

`.venv` 使用 `include-system-site-packages = false`，不会读取全局 Python 的第三方包。PyTorch 使用项目外的本地 wheel 安装，原因是该 wheel 体积较大，不适合提交到仓库；重建时先同步锁文件，再执行：

```powershell
uv pip install --python .venv\Scripts\python.exe `
  'D:\i-predictor\datasets\raw\torch-2.11.0+cu128-cp311-cp311-win_amd64.whl'
```

`.venv-system-site/` 是继承全局 Python 3.11 第三方包的旧环境，不参与统一测试。onnxruntime 与 donkeycar 已加入 `.venv`（2026-10-21 在 `.venv` 中实测 ONNX 评估、FastAPI/gRPC 推理、INT8 量化和驾驶 pilot 推理正常，未启动模拟器窗口），这些功能不再依赖该旧环境。

## CUDA 测试环境

`MyFlows/core/device.py` 在初始化 CUDA、导入 CuPy 前，优先将当前进程的 `CUDA_PATH` 指向 Python 环境内的 `nvidia/cuda_runtime`。这使 CUDA 12 NVRTC 使用 CUDA 12 头文件，避免系统 `CUDA_PATH=v13.3` 引入不兼容头文件。没有此 runtime wheel 时沿用原来的系统路径。此选择不修改 Windows 用户或系统环境变量。

正常框架入口无需手动设置环境变量。使用统一测试入口同时收集 unittest 类和函数式测试：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m tools.run_tests --scope all
```

CuPy 在导入时缓存 `CUDA_PATH`。先导入 CuPy、后导入 MyFlows 的进程（部分测试如此）由 `device.py` 把 CuPy 缓存的路径改回 wheel 内的 CUDA 12 头文件；否则 CuPy 磁盘缓存里没有的新 kernel 会按系统 CUDA 13 头文件编译并失败（已缓存的 kernel 会掩盖这个问题）。完全不导入 MyFlows 的独立 CuPy 脚本，仍应在新进程启动前指定 CUDA 12 头文件：

```powershell
$savedCudaPath = $env:CUDA_PATH
try {
    $env:CUDA_PATH = "$PWD\.venv\Lib\site-packages\nvidia\cuda_runtime"
    .\.venv\Scripts\python.exe -X utf8 -c "import cupy; cupy.show_config()"
} finally {
    $env:CUDA_PATH = $savedCudaPath
}
```

## 原生 CUDA 库构建

原生库由 `MyFlows/ops/cuda_native/build.py` 调用 CMake 3.20+ 构建，产物由同目录的 `native.py` 加载：

| 平台 | 编译器 | 默认 CMake 生成器 | 构建目录 | 产物 |
| --- | --- | --- | --- | --- |
| Windows | MSVC（Visual Studio 2022） | `Visual Studio 17 2022`（x64） | `build/native-cuda/build-msvc` | `build/native-cuda/native_cublas.dll` |
| Linux | GCC 或 Clang | 找得到 `ninja` 时为 `Ninja`，否则 `Unix Makefiles` | `build/native-cuda/build-ninja` 或 `build-unix-makefiles` | `build/native-cuda/native_cublas.so` |

不需要手动构建：第一次调用原生算子时，`native.py` 发现库不存在或与源码不一致（`native_cublas.cpp`、`CMakeLists.txt` 的哈希与上次构建时记录在 `build/native-cuda/native_cublas.*.sources` 里的不同），会先自动构建再加载。训练、测试和控制台任务都遵循这一规则，原生后端不会因为库缺失而改用 CuPy；构建失败时直接报错，CMake 输出在 `build/native-cuda/build.log`。构建期间持有 `build/native-cuda/build.lock` 文件锁，多个进程（分布式 Worker、并行测试）同时需要时只构建一次，其余进程等待后直接加载；分布式训练和分布式实验会在启动 Worker 之前先在主进程里构建好。设置 `MYFLOWS_NATIVE_AUTOBUILD=0` 可关闭自动构建，此时缺库直接报错。

也可以手动构建，脚本总是重新构建并输出 CMake 日志（加 `--if-needed` 则只在缺失或过期时构建）：

```powershell
uv run --no-sync python tools\build_native_cuda.py
```

`build/` 不纳入 Git，换机器或清理后会在第一次使用时重新构建。Windows 上如果有进程还占着旧的 DLL，覆盖会失败，结束该进程后重试即可。设置 `MF_NATIVE_CMAKE_GENERATOR` 可改用其他生成器，每种生成器有自己的构建目录（CMake 不允许用另一种生成器重新配置同一目录）。脚本先在 PATH 中找 `cmake`，再找解释器所在目录，因此也可以用 `uv pip install cmake ninja` 装进项目环境。

原生库只链接系统加载器，CUDA 驱动（Windows 为 `nvcuda.dll`，Linux 为 `libcuda.so.1`）、NVRTC 和 cuBLAS 都在运行时按路径加载，所以构建不需要 CUDA Toolkit。`native.py` 先用 `.venv` 里 nvidia wheel 的库：Windows 在各包的 `bin/` 下找 `cublas64_12.dll`、`nvrtc64_120_0.dll`，Linux 在 `lib/` 下找 `libcublas.so.12`、`libnvrtc.so.12`；找不到时再找 PyTorch 的 `torch/lib` 和 CUDA Toolkit（Windows 为 `CUDA_PATH\bin`；Linux 为 `CUDA_PATH`、`CUDA_HOME` 或 `/usr/local/cuda` 下的 `lib64/`）。实际加载的路径记录在 `native.info()` 的 `cublas_library`、`nvrtc_library` 字段中。

原生库在 `mf_init` 时用 NVRTC 编译全部 kernel：NVRTC 支持当前 GPU 的架构时直接生成该架构的 SASS（cubin），否则生成 PTX 交给驱动 JIT。结果按源码、编译选项与 NVRTC 版本的哈希缓存到 `build/native-cuda/kernel-cache/`（本机 RTX 4060 Laptop、sm_89：冷编译约 0.8 s，命中缓存约 24 ms）。缓存键与操作系统无关：同一份源码和 NVRTC 12.6 在 Windows 与 Linux 上算出同一个缓存文件名。设置 `MYFLOWS_NATIVE_KERNEL_CACHE=0` 可关闭缓存；设置 `MYFLOWS_NATIVE_SYNC=1`（或调用 `set_debug_sync(True)`）会在每次原生 launch 后同步，让异步错误报出具体 kernel 和源码行。

## Linux 环境

2026-10-23 在 WSL2 的 Ubuntu 22.04 中验证过原生库：GCC 11.4、CMake 4.4、Ninja 1.13，GPU 为 Windows 驱动 591.74 提供的 RTX 4060 Laptop。新建的 Python 3.11 环境只装了 numpy、cupy-cuda12x 和三个 nvidia wheel，没有 PyTorch 和 CUDA Toolkit。`Ninja` 与 `Unix Makefiles` 两种生成器都构建出了 `native_cublas.so`，20 项原生 kernel 测试全部通过。完整测试套件、ONNX Runtime GPU 推理、模拟器驾驶和 Nsight 采集尚未在 Linux 上实测。

需要准备：

- NVIDIA 驱动。WSL2 使用 Windows 端的驱动，由它提供 `/usr/lib/wsl/lib/libcuda.so.1`，不要在 WSL 里另装 Linux 驱动。
- Python 3.11（`uv` 可自动下载）、g++（`apt install build-essential`）和 CMake 3.20+。

```bash
uv sync --locked --inexact
.venv/bin/python tools/build_native_cuda.py   # 可省略：第一次用到原生算子时会自动构建
.venv/bin/python -X utf8 -m tools.run_tests --scope all
```

与 Windows 的差异：

- 解释器是 `.venv/bin/python`（Windows 为 `.venv\Scripts\python.exe`），路径分隔符用 `/`。不导入 MyFlows 的独立 CuPy 脚本这样指定 CUDA 12 头文件：

  ```bash
  CUDA_PATH="$PWD/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime" .venv/bin/python -X utf8 -c "import cupy; cupy.show_config()"
  ```

- 控制台前端：`(cd apps/console/web && npm ci && npm run build)`。
- PyTorch 在 Windows 上只用来给 ONNX Runtime 提供 cuDNN（`torch/lib`），原生库和 CuPy 都不需要它。ONNX Runtime 的 CUDA 推理在 Linux 上同样需要 cuDNN 9 等库，尚未配置。
- Nsight：`benchmark/nsight.py` 按以下顺序查找 `nsys` 和 `ncu`：命令行参数（`--tool-path`、`--ncu-exe`）、环境变量 `MF_NSYS` / `MF_NCU`、PATH，最后取默认安装目录中版本号最大的一个。Windows 的默认目录是 `C:\Program Files\NVIDIA Corporation\Nsight Systems <版本>\target-windows-x64\nsys.exe` 和 `Nsight Compute <版本>\target\windows-desktop-win7-x64\ncu.exe`，PATH 上的 `ncu.bat` 会换成它启动的 `ncu.exe`，因为经 cmd.exe 运行时 kernel 正则会在 `|` 处被截断。Linux 的默认目录是 `/opt/nvidia/nsight-systems/<版本>/bin/nsys`、`/opt/nvidia/nsight-compute/<版本>/ncu` 和 `/usr/local/cuda/bin`。
- 模拟器：`mycar/myconfig.py` 按 `sys.platform` 选择仓库根目录下的 `DonkeySimWin/donkey_sim.exe`、`DonkeySimLinux/donkey_sim.x86_64` 或 `DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim`，均从 gym-donkeycar 的发布页下载后解压，Linux 版需先 `chmod +x`。环境变量 `DONKEY_SIM_PATH` 可指定其他路径；设为 `remote` 时连接已经启动的模拟器。路径不存在时 gym-donkeycar 只提示手动启动模拟器。

macOS 没有 NVIDIA CUDA，只能运行 NumPy 的 CPU 路径；而且 cupy-cuda12x、nvidia-* 和 onnxruntime-gpu 没有 macOS wheel，`uv sync` 目前无法完成。
