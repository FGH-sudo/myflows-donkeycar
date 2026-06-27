# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python ML/DonkeyCar project. `MyFlows/` contains the custom deep learning framework and `MyFlows/tests/` test coverage. Application entry points live in `apps/`: shared utilities in `apps/common/`, training in `apps/train/`, evaluation in `apps/eval/`, and FastAPI/gRPC inference in `apps/serve/`. Utility scripts are in `tools/`, experiment helpers in `scripts/` and `benchmark/`, protobuf definitions in `proto/`, generated gRPC code in `generated/grpc/`, deployment manifests in `deploy/`, and design/report docs in `docs/`. Treat `mycar/data/`, `mycar/generated-road-data/`, and `DonkeySimWin/` as large runtime assets; avoid moving them casually.

## Build, Test, and Development Commands

- `pip install -r requirements-deploy.txt`: install serving and ONNX dependencies.
- `pip install -r MyFlows/requirements-tb.txt`: optional TensorBoard/PyTorch visualization support.
- `python -m unittest discover -s MyFlows/tests -p "test_*.py"`: run the framework test suite.
- `python -m tools.convert_generated_road_to_tub_v2 --src mycar/generated-road-data --dst mycar/data --clear-dst`: rebuild DonkeyCar tub data.
- `python -m apps.train.train_myflows_donkey --max-samples 200 --epochs 1 --device auto`: quick training smoke run.
- `python -m apps.serve.serve_fastapi --model mycar/models/myflow_resnet18_best.onnx --port 8000`: start local HTTP inference.
- `docker compose -f deploy/docker/docker-compose.yml up --build`: build and run the deployment stack.

Regenerate gRPC stubs after editing `proto/infer.proto` with `python -m grpc_tools.protoc -I proto --python_out=generated/grpc --grpc_python_out=generated/grpc proto/infer.proto`.

## Coding Style & Naming Conventions

Use Python 3 modules with explicit imports and CLI `main()` functions. Prefer 4-space indentation for new code, descriptive snake_case for functions and files, PascalCase for classes, and lowercase package directories. Keep generated files under `generated/`; do not hand-edit protobuf output except to verify imports. No formatter configuration is present, so match nearby style.

## Testing Guidelines

Tests use `unittest` and are named `test_*.py` under `MyFlows/tests/`. Add focused tests near the framework module being changed. For training, serving, or benchmark changes, include a small `--max-samples` smoke command in verification notes.

## Commit & Pull Request Guidelines

The current `main` branch has no commits, so there is no existing history convention. Use concise imperative subjects with an optional scope, for example `apps/train: add resume smoke check`. Pull requests should describe the change, list commands run, call out dataset/model artifacts touched, and include screenshots or metric tables when changing visualizations, reports, or benchmark outputs.

## Security & Configuration Tips

Do not commit local credentials, large regenerated model checkpoints, or machine-specific CUDA paths. Keep deployment defaults in `deploy/` and runtime DonkeyCar settings in `mycar/config.py` or `mycar/myconfig.py`.
