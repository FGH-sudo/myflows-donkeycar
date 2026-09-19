"""Generate distributed PS/Ring protobuf stubs into generated/grpc."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated" / "grpc"
PROTOS = [ROOT / "proto" / "common.proto", ROOT / "proto" / "ps.proto", ROOT / "proto" / "ring.proto"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "__init__.py").touch()
    command = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"-I{ROOT / 'proto'}",
        f"--python_out={OUT}",
        f"--grpc_python_out={OUT}",
        *[str(path) for path in PROTOS],
    ]
    subprocess.check_call(command)
    replacements = {
        "import common_pb2 as": "from generated.grpc import common_pb2 as",
        "import ps_pb2 as": "from generated.grpc import ps_pb2 as",
        "import ring_pb2 as": "from generated.grpc import ring_pb2 as",
    }
    for path in OUT.glob("*_pb2*.py"):
        text = path.read_text(encoding="utf-8")
        for old, new in replacements.items():
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")
    print(f"generated stubs in {OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
