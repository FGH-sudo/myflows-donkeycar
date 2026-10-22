# 开发环境

本项目的原生 CUDA 路径和 Python 实验路径分开管理：系统 CUDA Toolkit 负责 `nvcc`、MSVC/CMake 构建和 Nsight；隔离的 `.venv` 负责 CuPy、PyTorch 与课程项目依赖。CuPy 需要的 CUDA 12.6 运行库和 NVRTC 头文件也固定安装在 `.venv` 内，避免被系统 Toolkit 13.3 的头文件覆盖。

## 已安装工具

- NVIDIA CUDA Toolkit 13.3.1，`nvcc` 版本 13.3.73。
- Visual Studio Community 2022 17.14.27，MSVC 14.44.35207，Windows SDK 10.0.26100.0。
- CMake 4.4.3。
- Nsight Compute 2026.2.1，命令行程序位于 `C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\target\windows-desktop-win7-x64\ncu.exe`。
- `uv` 0.11.9。

## Python 依赖

根目录的 `pyproject.toml` 和 `uv.lock` 是项目 Python 依赖的声明与锁定文件。当前使用 `cupy-cuda12x==14.0.1`，对应 CUDA 12 系列；`.venv` 内固定安装 `nvidia-cublas-cu12`、`nvidia-cuda-nvrtc-cu12` 和 `nvidia-cuda-runtime-cu12`。PyTorch wheel 自带同名的 `nvrtc64_120_0.dll`（12.8）：进程里先加载哪一个，CuPy 就用哪一个，而 12.8 的 NVRTC 配 12.6 的头文件会让 CUB kernel（如布尔规约）编译失败。因此 `MyFlows/core/device.py` 只用 `importlib.util.find_spec` 定位 `torch/lib`（供 ONNX Runtime 加载 cuDNN），不再导入 torch；原生 DLL 也优先加载 `.venv` 内的 NVRTC 12.6 与 cuBLAS。实际版本可由 `MyFlows.ops.cuda_native.info()` 读出（`nvrtc`、`binary`、`device_arch` 等字段），实验应记录该输出而不是只看包版本。

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

原生 DLL 构建仍使用现有 CMake/MSVC 入口：

```powershell
uv run --no-sync python tools\build_native_cuda.py
```

该脚本默认使用 `Visual Studio 17 2022` 的 x64 生成器，构建目录为 `build/native-cuda/build-msvc`，产物 `build/native-cuda/native_cublas.dll` 由 `MyFlows/ops/cuda_native/native.py` 加载；`build/` 不纳入 Git，换机器或清理后重新执行上面的命令即可。如果需要使用其他 CMake 生成器，可设置 `MF_NATIVE_CMAKE_GENERATOR`。

DLL 在 `mf_init` 时用 NVRTC 编译全部 kernel：NVRTC 支持当前 GPU 的架构时直接生成该架构的 SASS（cubin），否则生成 PTX 交给驱动 JIT。结果按源码、编译选项与 NVRTC 版本的哈希缓存到 `build/native-cuda/kernel-cache/`（本机 RTX 4060 Laptop、sm_89：冷编译约 0.8 s，命中缓存约 24 ms）。设置 `MYFLOWS_NATIVE_KERNEL_CACHE=0` 可关闭缓存；设置 `MYFLOWS_NATIVE_SYNC=1`（或调用 `set_debug_sync(True)`）会在每次原生 launch 后同步，让异步错误报出具体 kernel 和源码行。
