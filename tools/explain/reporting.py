# -*- coding: utf-8 -*-
"""Grad-CAM 输出报告工具。"""

from __future__ import annotations

from pathlib import Path


def _display_path(path: Path | None, root: Path | None = None) -> str:
    if path is None:
        return ""
    p = Path(path)
    if root is not None:
        try:
            p = p.resolve().relative_to(Path(root).resolve())
        except ValueError:
            pass
    return p.as_posix()


def gradcam_report_header(
    model_type: str,
    checkpoint: Path,
    layer_name: str,
    logdir: Path,
    *,
    split: str = "all",
    split_file: Path | None = None,
    fixed_throttle: float = 0.5,
    force_fixed_throttle: bool = False,
    sample_count: int | None = None,
    root: Path | None = None,
) -> list[str]:
    split_text = (
        f"`{split}` via `{_display_path(split_file, root)}`" if split_file else f"`{split}`"
    )
    fixed_text = f"`{fixed_throttle}` forced" if force_fixed_throttle else f"`{fixed_throttle}` fallback"
    return [
        "# Grad-CAM 可视化报告",
        "",
        f"- model_type: `{model_type}`",
        f"- checkpoint: `{_display_path(checkpoint, root)}`",
        f"- layer: `{layer_name}`",
        f"- split: {split_text}",
        f"- fixed_throttle: {fixed_text}",
        f"- samples: `{sample_count}`" if sample_count is not None else "- samples: `unknown`",
        f"- TensorBoard: `{_display_path(logdir, root)}`",
        "- score 是 target_output 对应的模型原始预测值，不是准确率或置信度。",
        "",
        "| # | image | target_output | score | true_angle | pred_angle | abs_error | overlay |",
        "|---|-------|---------------|-------|------------|------------|-----------|---------|",
    ]


def append_gradcam_row(
    lines: list[str],
    *,
    step: int,
    rel_path: Path,
    target_label: str,
    score: float,
    true_angle: float,
    pred_angle: float,
    abs_error: float,
    overlay_path: Path,
    root: Path,
) -> None:
    lines.append(
        f"| {step} | `{_display_path(rel_path)}` | `{target_label}` | {score:.6f} | "
        f"{true_angle:.6f} | {pred_angle:.6f} | {abs_error:.6f} | "
        f"`{_display_path(overlay_path, root)}` |"
    )


def write_report(lines: list[str], out_dir: Path) -> Path:
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
