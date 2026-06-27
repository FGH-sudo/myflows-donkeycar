# -*- coding: utf-8 -*-
"""Fixed-speed MyFlows ResNet-18 pilot for DonkeyCar."""

from pathlib import Path
import os
import sys
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import MyFlows as ms  # noqa: E402
from tools.device_runtime import create_ort_inference_session, print_ort_device, resolve_myflows_device


class MyFlowsResNet18Pilot:
    def __init__(
        self,
        checkpoint_path,
        image_w=160,
        image_h=120,
        fixed_throttle=0.5,
        max_throttle=0.5,
        steering_scale=1.0,
        debug=False,
        device="auto",
    ):
        checkpoint = Path(checkpoint_path).resolve()
        self.device = str(device)
        self.checkpoint_path = str(checkpoint)
        self.ckpt_stem = checkpoint.with_suffix("") if checkpoint.suffix == ".onnx" else checkpoint
        self.image_w = int(image_w)
        self.image_h = int(image_h)
        self.max_throttle = abs(float(max_throttle))
        self.fixed_throttle = float(np.clip(float(fixed_throttle), 0.0, self.max_throttle))
        self.steering_scale = float(steering_scale)
        self.debug = bool(debug)
        self._last_print = 0.0
        self.session = None
        self.input_name = None
        self.input_dtype = np.float32

        print(
            f"[MyFlowsPilot] fixed throttle={self.fixed_throttle:.3f}; "
            "model output is used for steering only"
        )

        onnx_path = checkpoint if checkpoint.suffix == ".onnx" else checkpoint.with_suffix(".onnx")
        if onnx_path.exists():
            try:
                import onnxruntime as ort

                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 1)
                sess_options.inter_op_num_threads = 1
                self.session, providers = create_ort_inference_session(
                    onnx_path, self.device, sess_options=sess_options
                )
                print_ort_device(providers, self.device)
                self.input_name = self.session.get_inputs()[0].name
                input_type = self.session.get_inputs()[0].type
                self.input_dtype = np.float32 if "float" in input_type and "double" not in input_type else np.float64
                print(f"[MyFlowsPilot] loaded ONNX {onnx_path}")
                return
            except Exception as exc:
                print(f"[MyFlowsPilot] ONNX load failed, fallback to MyFlows: {exc}")

        resolve_myflows_device(self.device)
        self.x = ms.Variable(
            np.zeros((1, 3, self.image_h, self.image_w), dtype=np.float32),
            name="X",
        )
        self.model = ms.ResNet18(
            in_channels=3,
            num_classes=2,
            stem="cifar",
            base_width=64,
            name="resnet18_donkey",
        )
        self.out = self.model(self.x)
        self.graph = ms.Graph(self.out)
        ms.load_checkpoint([self.model], None, str(self.ckpt_stem))
        self.model.eval()
        self.graph.forward()
        print(f"[MyFlowsPilot] loaded JSON+NPZ {self.ckpt_stem}")

    def _preprocess(self, img_arr):
        if img_arr is None:
            return None
        arr = np.asarray(img_arr)
        if arr.ndim != 3:
            return None
        if arr.shape[0] != self.image_h or arr.shape[1] != self.image_w:
            arr = cv2.resize(arr, (self.image_w, self.image_h), interpolation=cv2.INTER_LINEAR)
        x = arr.astype(self.input_dtype) / 255.0
        return np.transpose(x, (2, 0, 1))[np.newaxis, ...]

    def run(self, img_arr):
        x = self._preprocess(img_arr)
        if x is None:
            return 0.0, 0.0

        start = time.time()
        if self.session is not None:
            pred = np.asarray(self.session.run(None, {self.input_name: x})[0]).reshape(-1)
        else:
            self.x.value = x.astype(np.float32, copy=False)
            self.model.eval()
            self.graph.forward()
            pred = np.asarray(self.out.value).reshape(-1)
        elapsed = time.time() - start

        angle = float(np.clip(float(pred[0]) * self.steering_scale, -1.0, 1.0))
        throttle = self.fixed_throttle

        now = time.time()
        if self.debug and now - self._last_print > 1.0:
            print(
                f"[MyFlowsPilot] angle={angle: .3f} throttle={throttle: .3f} "
                f"infer={elapsed:.3f}s raw_angle={float(pred[0]): .3f} "
                f"steering_scale={self.steering_scale:.2f}"
            )
            self._last_print = now
        return angle, throttle
