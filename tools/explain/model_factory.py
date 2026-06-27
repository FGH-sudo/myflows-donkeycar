# -*- coding: utf-8 -*-
"""Grad-CAM 模型构建与特征层选择。"""

from __future__ import annotations

import numpy as np

import MyFlows as ms


def build_gradcam_model(model_type: str, h: int, w: int, num_classes: int, dtype) -> tuple[object, object, object]:
    x_var = ms.Variable(np.zeros((1, 3, h, w), dtype=dtype), name="X")
    if model_type == "resnet":
        model = ms.ResNet18(in_channels=3, num_classes=2, stem="cifar", base_width=64, name="resnet18_donkey")
    elif model_type == "vgg":
        model = ms.VGG11(in_channels=3, num_classes=num_classes, image_h=h, image_w=w, name="vgg11_donkey")
    else:
        raise ValueError(f"unsupported model_type={model_type!r}")
    logits = model(x_var)
    return x_var, model, logits


def select_feature_node(model, preferred: str | None):
    feature_nodes = getattr(model, "_last_feature_nodes", {})
    if not feature_nodes:
        raise RuntimeError("model did not expose feature nodes for Grad-CAM")
    if preferred and preferred != "last":
        if preferred not in feature_nodes:
            raise SystemExit(f"Grad-CAM layer not found: {preferred}; available={list(feature_nodes)}")
        return preferred, feature_nodes[preferred]
    name = "layer4" if "layer4" in feature_nodes else next(reversed(feature_nodes))
    return name, feature_nodes[name]
