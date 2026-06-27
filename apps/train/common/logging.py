# -*- coding: utf-8 -*-
"""训练日志工具。"""

from __future__ import annotations

import sys
from pathlib import Path


class TeeLogger:
    """把 stdout 同时写到终端和日志文件。"""

    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.stderr = sys.stderr
        self.file = Path(log_path).open("a", encoding="utf-8", buffering=1)

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.file.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        if not self.file.closed:
            self.file.flush()

    def close(self) -> None:
        if sys.stdout is self:
            sys.stdout = self.terminal
        if sys.stderr is self:
            sys.stderr = self.stderr
        if not self.file.closed:
            self.file.close()
