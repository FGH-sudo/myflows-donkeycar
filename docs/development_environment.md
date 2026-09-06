# 开发环境

本项目的原生 CUDA 路径和 Python 实验路径分开管理：系统 CUDA Toolkit 负责 `nvcc`、MSVC/CMake 构建和 Nsight；隔离的 `.venv` 负责 CuPy、PyTorch 与课程项目依赖。CuPy 需要的 CUDA 12.6 运行库和 NVRTC 头文件也固定安装在 `.venv` 内，避免被系统 Toolkit 13.3 的头文件覆盖。

## 已安装工具

- NVIDIA CUDA Toolkit 13.3.1，`nvcc` 版本 13.3.73。
- Visual Studio Community 2022 17.14.27，MSVC 14.44.35207，Windows SDK 10.0.26100.0。
- CMake 4.4.3。
- Nsight Compute 2026.2.1，命令行程序位于 `C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\target\windows-desktop-win7-x64\ncu.exe`。
- `uv` 0.11.9。

## Python 依赖

根目录的 `pyproject.toml` 和 `uv.lock` 是阶段一 Python 依赖的声明与锁定文件。当前使用 `cupy-cuda12x==14.0.1`，因为 CUDA 13.3 Toolkit 提供的是 cuBLAS 12 ABI，而 `cupy-cuda13x` 轮子要求 `cublas64_13.dll`。系统 Toolkit 仍用于原生 C++/CUDA 调度 DLL；CuPy 的 CUDA 12.6 运行库由 `.venv` 中固定版本的 `nvidia-cublas-cu12`、`nvidia-cuda-nvrtc-cu12` 和 `nvidia-cuda-runtime-cu12` 提供。

同步阶段一依赖时使用：

```powershell
uv sync
uv pip check --python .venv\Scripts\python.exe
```

`.venv` 使用 `include-system-site-packages = false`，不会读取全局 Python 的第三方包。PyTorch 使用项目外的本地 wheel 安装，原因是该 wheel 体积较大，不适合提交到仓库；重建时执行：

```powershell
uv pip install --python .venv\Scripts\python.exe `
  'D:\i-predictor\datasets\raw\torch-2.11.0+cu128-cp311-cp311-win_amd64.whl'
```

历史环境仍保留在 `.venv-system-site/`，仅用于回溯，不参与当前测试。

## CUDA 测试环境

为了让 CuPy 使用现有 PyTorch 的 CUDA 12 运行库，运行 Python 测试前在当前 PowerShell 设置：

```powershell
$runtime = "$PWD\.venv\Lib\site-packages\nvidia\cuda_runtime"
$nvrtc = "$PWD\.venv\Lib\site-packages\nvidia\cuda_nvrtc"
$cublas = "$PWD\.venv\Lib\site-packages\nvidia\cublas"
$torchLib = "$PWD\.venv\Lib\site-packages\torch\lib"
$env:CUDA_PATH = $runtime
$env:CUPY_CUDA_PATH = $runtime
$env:PATH = "$cublas\bin;$nvrtc\bin;$runtime\bin;$torchLib;$env:PATH"
uv run --no-sync python -m unittest discover -s MyFlows/tests -p 'test_*.py'
```

原生 DLL 使用完整 Toolkit 和 MSVC 构建，不依赖上面临时设置的 `CUDA_PATH`：

```powershell
uv run --no-sync python tools\build_native_cuda.py
```

该脚本默认使用 `Visual Studio 17 2022` 的 x64 生成器，构建目录为 `.codex/native-cuda/build-msvc`。如果需要使用其他 CMake 生成器，可设置 `MF_NATIVE_CMAKE_GENERATOR`。
