# 开发环境

本项目的原生 CUDA 路径和 Python 实验路径分开管理：系统 CUDA Toolkit 负责 `nvcc`、MSVC/CMake 构建和 Nsight；隔离的 `.venv` 负责 CuPy、PyTorch 与课程项目依赖。CuPy 需要的 CUDA 12.6 运行库和 NVRTC 头文件也固定安装在 `.venv` 内，避免被系统 Toolkit 13.3 的头文件覆盖。

## 已安装工具

- NVIDIA CUDA Toolkit 13.3.1，`nvcc` 版本 13.3.73。
- Visual Studio Community 2022 17.14.27，MSVC 14.44.35207，Windows SDK 10.0.26100.0。
- CMake 4.4.3。
- Nsight Compute 2026.2.1，命令行程序位于 `C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\target\windows-desktop-win7-x64\ncu.exe`。
- `uv` 0.11.9。

## Python 依赖

根目录的 `pyproject.toml` 和 `uv.lock` 是阶段一 Python 依赖的声明与锁定文件。当前使用 `cupy-cuda12x==14.0.1`，对应 CUDA 12 系列；`.venv` 内固定安装 `nvidia-cublas-cu12`、`nvidia-cuda-nvrtc-cu12` 和 `nvidia-cuda-runtime-cu12`。框架也会加载 PyTorch 自带的 CUDA DLL，因此实际 NVRTC 版本受导入顺序影响，应在同一运行进程记录，不能只根据包版本推断。

同步阶段一依赖时使用：

```powershell
uv sync --locked --inexact
uv pip check --python .venv\Scripts\python.exe
```

默认 `dev` 依赖组包括统一测试入口需要的 FastAPI、httpx 和 python-multipart，并限定 `setuptools<82`，与当前 PyTorch wheel 的依赖约束一致。`--inexact` 保留本地 wheel 安装的 PyTorch 及其额外依赖；普通的精确 `uv sync` 会移除未声明在锁文件中的包。

`.venv` 使用 `include-system-site-packages = false`，不会读取全局 Python 的第三方包。PyTorch 使用项目外的本地 wheel 安装，原因是该 wheel 体积较大，不适合提交到仓库；重建时先同步锁文件，再执行：

```powershell
uv pip install --python .venv\Scripts\python.exe `
  'D:\i-predictor\datasets\raw\torch-2.11.0+cu128-cp311-cp311-win_amd64.whl'
```

历史环境仍保留在 `.venv-system-site/`，仅用于回溯，不参与当前测试。

## CUDA 测试环境

`MyFlows/core/device.py` 在初始化 CUDA、导入 CuPy 前，优先将当前进程的 `CUDA_PATH` 指向 Python 环境内的 `nvidia/cuda_runtime`。这使 CUDA 12 NVRTC 使用 CUDA 12 头文件，避免系统 `CUDA_PATH=v13.3` 引入不兼容头文件。没有此 runtime wheel 时沿用原来的系统路径。此选择不修改 Windows 用户或系统环境变量。

正常框架入口无需手动设置环境变量。使用统一测试入口同时收集 unittest 类和函数式测试：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m tools.run_tests --scope all
```

如果运行独立 CuPy 脚本（不经过框架的 CUDA 初始化，或先导入 CuPy），应在新进程启动前指定 CUDA 12 头文件；CuPy 会缓存已检测到的路径：

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

该脚本默认使用 `Visual Studio 17 2022` 的 x64 生成器，构建目录为 `.codex/native-cuda/build-msvc`。如果需要使用其他 CMake 生成器，可设置 `MF_NATIVE_CMAKE_GENERATOR`。
