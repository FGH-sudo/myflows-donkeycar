# -*- coding: utf-8 -*-
"""Grad-CAM 输出报告工具。"""

from __future__ import annotations

from pathlib import Path


def gradcam_report_header(model_type: str, checkpoint: Path, layer_name: str, logdir: Path) -> list[str]:
    return [
        "# Grad-CAM 可视化报告",
        "",
        f"- model_type: `{model_type}`",
        f"- checkpoint: `{checkpoint}`",
        f"- layer: `{layer_name}`",
        f"- TensorBoard: `{logdir}`",
        "",
        "| # | image | target | score | overlay |",
        "|---|-------|--------|-------|---------|",
    ]


def append_gradcam_row(
    lines: list[str],
    *,
    step: int,
    rel_path: Path,
    target_label: str,
    score: float,
    overlay_path: Path,
    root: Path,
) -> None:
    lines.append(
        f"| {step} | `{rel_path}` | `{target_label}` | {score:.6f} | `{overlay_path.relative_to(root)}` |"
    )


def write_report(lines: list[str], out_dir: Path) -> Path:
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
