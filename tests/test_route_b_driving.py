# -*- coding: utf-8 -*-
"""Route B driving tests: unified ROI crop parity, trainer recipe & resume, console presets, RoadFinder bimodal."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from apps.collect.vision import RoadFinder
from apps.common.image_preprocess import crop_rgb, format_roi_crop, imread_nchw, parse_roi_crop
from apps.console.presets import ValidationError, validate_single
from apps.serve.onnx_predictor import OnnxPredictor
from apps.train import train_myflows_donkey as train_mod
from MyFlows.utils.checkpoint import list_training_states
from mycar.myflows_pilot import MyFlowsResNet18Pilot
from tools.export_resnet_onnx import export_resnet_onnx

ROOT = Path(__file__).resolve().parents[1]


def _make_road_tub(root: Path, count: int = 16, *, height: int = 64, width: int = 96) -> Path:
    tub = root / "tub"
    images = tub / "images"
    images.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        angle = -0.7 + 1.4 * (idx / max(1, count - 1))
        img = np.full((height, width, 3), 45, dtype=np.uint8)
        cx = int(np.clip(width * 0.5 + angle * width * 0.25, 6, width - 6))
        img[height // 3 :, cx - 3 : cx + 3] = (200, 210, 220)
        cv2.imwrite(str(images / f"{idx}_{angle:.4f}.jpg"), img)
    return tub


def _run_train(work: Path, name: str, *extra: str) -> tuple[subprocess.CompletedProcess, Path]:
    out = work / name
    cmd = [
        sys.executable, "-X", "utf8", "-m", "apps.train.train_myflows_donkey",
        "--data", str(work / "tub"), "--catalog", "missing.catalog", "--device", "cpu",
        "--base-width", "4", "--batch", "4", "--epochs", "2", "--val-size", "4",
        "--seed", "7", "--no-tensorboard", "--checkpoint-every", "0",
        "--log-dir", str(out), "--out", str(out / "ckpt"), "--best-out", str(out / "best"),
        "--stop-file", str(out / "STOP"), "--run-dir", str(out / "run"), "--sync-timeout", "60",
        *extra,
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    return proc, out


def _ckpt_params(out: Path) -> dict:
    with np.load(out / "ckpt.npz") as archive:
        return {k: archive[k] for k in archive.files if k.startswith(("param_", "buffer_"))}


class RoiCropAndServeParityTest(unittest.TestCase):
    def test_parse_format_and_crop_rgb(self) -> None:
        self.assertIsNone(parse_roi_crop(None))
        self.assertIsNone(parse_roi_crop(""))
        self.assertIsNone(parse_roi_crop("none"))
        self.assertIsNone(parse_roi_crop("0,0"))
        self.assertEqual(parse_roi_crop("60, 16"), (60, 16, 0, 0))
        self.assertEqual(parse_roi_crop("60,16,8,8"), (60, 16, 8, 8))
        self.assertEqual(parse_roi_crop((60, 16)), (60, 16, 0, 0))
        self.assertEqual(format_roi_crop((60, 16, 0, 0)), "60,16")
        self.assertEqual(format_roi_crop((60, 16, 4, 8)), "60,16,4,8")
        self.assertIsNone(format_roi_crop(None))
        for bad in ("60", "60,16,0", "-1,10", "60,bad", (True, 1)):
            with self.assertRaises(ValueError):
                parse_roi_crop(bad)

        arr = np.zeros((160, 320, 3), dtype=np.uint8)
        cropped = crop_rgb(arr, (60, 16, 10, 10))
        self.assertEqual(cropped.shape, (84, 300, 3))
        with self.assertRaises(ValueError):
            crop_rgb(arr, (100, 60, 0, 0))

    def test_imread_nchw_roi_crop_matches_onnx_predictor_pilot_and_eval(self) -> None:
        import MyFlows as ms
        from MyFlows.core.device import set_device
        from MyFlows.core.dtype import set_default_dtype
        from apps.train.common.regression_model import build_regression_resnet

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            images = work / "tub" / "images"
            images.mkdir(parents=True)
            rng = np.random.default_rng(42)
            rgb_src = rng.integers(0, 255, size=(160, 320, 3), dtype=np.uint8)
            img_path = images / "0_0.1500.jpg"
            cv2.imwrite(str(img_path), cv2.cvtColor(rgb_src, cv2.COLOR_RGB2BGR))
            img_path2 = images / "1_-0.2500.jpg"
            cv2.imwrite(str(img_path2), cv2.cvtColor(rgb_src, cv2.COLOR_RGB2BGR))
            rgb = cv2.cvtColor(cv2.imread(str(img_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            ckpt_stem = work / "model"
            try:
                set_device("cpu")
                set_default_dtype(np.float32)
                x = ms.Variable(np.zeros((1, 3, 48, 80), dtype=np.float32), name="X", requires_grad=False)
                model = build_regression_resnet(
                    "resnet18", "global", 48, 80, in_channels=3, output_dim=2, stem="cifar",
                    base_width=4, name="resnet18_donkey",
                )
                model.fc_W.value *= 0.05
                out = model(x)
                model.eval()
                ms.Graph(out).forward()
                ms.save_checkpoint([model], None, 0, filepath=str(ckpt_stem), loss=0.1)
            finally:
                set_device("cpu")
                set_default_dtype(np.float64)

            onnx_path = export_resnet_onnx(
                ckpt_stem, batch_size=1, image_w=80, image_h=48, base_width=4, device="cpu", roi_crop="60,16"
            )
            predictor = OnnxPredictor(onnx_path, 80, 48, device="cpu")
            self.assertEqual(predictor.roi_crop, (60, 16, 0, 0))
            self.assertEqual(predictor.info()["roi_crop"], "60,16")

            direct_in = imread_nchw(img_path, (80, 48), dtype=np.float32, roi_crop=(60, 16, 0, 0))
            direct_pred = np.asarray(predictor.session.run(None, {predictor.input_name: direct_in})[0]).reshape(-1)
            pred_serve = predictor.predict(rgb)
            np.testing.assert_allclose(pred_serve, direct_pred, rtol=1e-5, atol=1e-5)

            pilot_onnx = MyFlowsResNet18Pilot(onnx_path, image_w=80, image_h=48, device="cpu", prefer_onnx=True)
            pilot_npz = MyFlowsResNet18Pilot(
                ckpt_stem, image_w=80, image_h=48, device="cpu", prefer_onnx=False,
                training_config={"args": {"base_width": 4, "stem": "cifar", "roi_crop": "60,16"}},
            )
            angle_onnx, _ = pilot_onnx.run(rgb)
            angle_npz, _ = pilot_npz.run(rgb)
            expected_clipped = float(np.clip(direct_pred[0], -1.0, 1.0))
            self.assertAlmostEqual(angle_onnx, expected_clipped, places=5)
            self.assertAlmostEqual(angle_npz, expected_clipped, places=5)
            self.assertAlmostEqual(angle_onnx, angle_npz, places=5)
            np.testing.assert_allclose(pilot_npz._preprocess(rgb), direct_in, rtol=1e-6, atol=1e-6)

            report_json = work / "onnx_eval.json"
            proc = subprocess.run(
                [
                    sys.executable, "-X", "utf8", "-m", "apps.eval.eval_myflows_donkey_onnx",
                    "--data", str(work / "tub"), "--catalog", "missing.catalog",
                    "--checkpoint", str(onnx_path), "--device", "cpu", "--report-json", str(report_json),
                ],
                cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            report = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertEqual(report["results"][0]["roi_crop"], "60,16")
            expected_mae = float((abs(direct_pred[0] - 0.15) + abs(direct_pred[0] + 0.25)) / 2.0)
            self.assertAlmostEqual(report["results"][0]["angle_mae"], expected_mae, places=5)


class RouteBTrainerAndResumeTest(unittest.TestCase):
    def test_route_b_recipe_export_and_bitwise_exact_resume(self) -> None:
        import onnxruntime as ort

        recipe = (
            "--roi-crop", "16,8",
            "--image-size", "32x64",
            "--lr", "0.002",
            "--lr-schedule", "cosine",
            "--warmup-epochs", "0.5",
            "--min-lr", "0.0001",
            "--loss", "huber",
            "--huber-delta", "0.2",
            "--angle-weight", "1.0",
            "--throttle-weight", "0.1",
            "--turn-weight", "3.0",
            "--best-metric", "val_angle_mae",
            "--augment-preset", "driving",
            "--state-every", "2",
            "--keep-states", "20",
            "--export-onnx",
        )
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            _make_road_tub(work, count=16, height=64, width=96)
            proc, full = _run_train(work, "full", *recipe)
            self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-3000:])

            result = json.loads((full / "run" / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["config"]["roi_crop"], "16,8")
            self.assertEqual(result["config"]["lr_schedule"], "cosine")
            self.assertEqual(result["config"]["loss"], "huber")
            self.assertEqual(result["config"]["augment_preset"], "driving")
            self.assertEqual(result["config"]["best_metric"], "val_angle_mae")
            self.assertEqual(result["metrics"]["best_metric"], "val_angle_mae")

            onnx_path = full / "best.onnx"
            self.assertTrue(onnx_path.is_file())
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            self.assertEqual(sess.get_modelmeta().custom_metadata_map["myflows.roi_crop"], "16,8")
            self.assertEqual(sess.get_inputs()[0].shape, [1, 3, 32, 64])

            states = list_training_states(full / "run" / "states")
            mid = next(s for s in states if not s["state"]["epoch_complete"] and s["step"] > 0)
            self.assertEqual(len(mid["state"]["rng"]["transforms"]), 3)
            self.assertEqual(mid["state"]["scheduler"]["type"], "cosine_warmup")

            proc_res, resumed = _run_train(work, "resumed", *recipe, "--resume-state", str(mid["path"]))
            self.assertEqual(proc_res.returncode, 0, proc_res.stdout[-3000:] + proc_res.stderr[-3000:])

            full_params = _ckpt_params(full)
            res_params = _ckpt_params(resumed)
            for k, v in full_params.items():
                np.testing.assert_array_equal(v, res_params[k], err_msg=k)
            res_result = json.loads((resumed / "run" / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(res_result["metrics"]["val_angle_mae"], result["metrics"]["val_angle_mae"])

            proc_half, half = _run_train(
                work, "half",
                *(a for a in recipe if a != "0.002" and a != "--lr"),
                "--lr", "0.001", "--resume-state", str(mid["path"]),
            )
            self.assertEqual(proc_half.returncode, 0, proc_half.stdout[-3000:] + proc_half.stderr[-3000:])
            half_res = json.loads((half / "run" / "result.json").read_text(encoding="utf-8"))
            self.assertEqual((half_res["resume"]["lr_before"], half_res["resume"]["lr"]), (0.002, 0.001))

    def test_mobilenet_v2_training_and_onnx_export(self) -> None:
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            _make_road_tub(work, count=8, height=64, width=96)
            out = work / "mnv2"
            cmd = [
                sys.executable, "-X", "utf8", "-m", "apps.train.train_myflows_donkey",
                "--data", str(work / "tub"), "--catalog", "missing.catalog", "--device", "cpu",
                "--arch", "mobilenet_v2", "--roi-crop", "12,4", "--image-size", "48x64",
                "--batch", "2", "--epochs", "1", "--val-size", "2", "--seed", "3", "--no-tensorboard",
                "--checkpoint-every", "0", "--log-dir", str(out), "--out", str(out / "ckpt"),
                "--best-out", str(out / "best"), "--stop-file", str(out / "STOP"), "--run-dir", str(out / "run"),
                "--export-onnx",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
            self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-3000:])
            onnx_path = out / "best.onnx"
            self.assertTrue(onnx_path.is_file())
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            self.assertEqual(sess.get_modelmeta().custom_metadata_map["myflows.roi_crop"], "12,4")
            self.assertEqual(sess.get_inputs()[0].shape, [1, 3, 48, 64])


class RouteBConsolePresetsTest(unittest.TestCase):
    def test_validate_single_route_b_args_and_fingerprint_preservation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mycar" / "data" / "images").mkdir(parents=True)
            validated = validate_single({
                "roi_crop": "60, 16, 0, 0",
                "lr": 0.001,
                "lr_schedule": "cosine",
                "warmup_epochs": 1.0,
                "min_lr": 1e-5,
                "loss": "huber",
                "huber_delta": 0.15,
                "angle_weight": 1.0,
                "throttle_weight": 0.1,
                "turn_weight": 3.0,
                "best_metric": "val_angle_mae",
                "augment_preset": "driving",
            }, root)
            self.assertEqual(validated["roi_crop"], "60,16")
            self.assertEqual(validated["best_metric"], "val_angle_mae")
            self.assertNotIn("angle_weight", validated)  # default 1.0 omitted

            defaults_stripped = validate_single({
                "roi_crop": "0,0",
                "lr_schedule": "constant",
                "warmup_epochs": 0.0,
                "min_lr": 0.0,
                "loss": "mse",
                "huber_delta": 0.2,
                "throttle_weight": 1.0,
                "turn_weight": 0.0,
                "best_metric": "val_loss",
                "augment_preset": "default",
            }, root)
            self.assertEqual(defaults_stripped, {"distributed": "none"})

            for bad in (
                {"roi_crop": "-5,10"},
                {"roi_crop": "bad"},
                {"lr": 0.001, "lr_schedule": "cosine", "min_lr": 0.01},
                {"loss": "smooth_l1"},
                {"huber_delta": 0.0},
                {"angle_weight": 0.0, "throttle_weight": 0.0},
                {"best_metric": "unknown"},
                {"augment_preset": "custom"},
            ):
                with self.subTest(bad=bad), self.assertRaises(ValidationError):
                    validate_single(bad, root)


class RoadFinderBimodalConsensusTest(unittest.TestCase):
    @staticmethod
    def _synthetic_bimodal_frame(*, glare_bimodal: bool = False) -> np.ndarray:
        """Render a 160x320 road where glare intermittently washes out yellow lane lines onto outer green grass."""
        h, w = 160, 320
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:73] = (140, 185, 225)
        img[73:] = (65, 130, 55)  # green grass verge outside curbs
        for row in range(78, h - 10):
            depth = float(row - 73)
            c0 = 160.0
            left_x = int(round(c0 - 2.6 * depth))
            right_lane_x = int(round(c0 + 2.6 * depth))
            right_curb_x = int(round(c0 + 5.8 * depth))
            # Road asphalt extends beyond lane line on sunny bridge exit before green grass starts.
            asphalt_l = max(0, left_x)
            asphalt_r = min(w, right_curb_x if glare_bimodal else right_lane_x + 4)
            img[row, asphalt_l:asphalt_r] = (128, 128, 130)
            # White left lane line
            if 0 <= left_x < w - 2:
                img[row, max(0, left_x - 2) : left_x + 1] = (240, 240, 240)
            # Yellow right lane line (washed out on alternating bands under glare)
            wash = glare_bimodal and ((row // 4) % 2 == 1)
            if not wash and 0 <= right_lane_x < w - 3:
                img[row, right_lane_x : min(w, right_lane_x + 3)] = (230, 195, 65)
        img[h - 12 :] = (90, 90, 95)
        return img

    def test_bimodal_consensus_recovers_true_lane_line_and_rejects_offroad(self) -> None:
        finder = RoadFinder()
        clean = self._synthetic_bimodal_frame(glare_bimodal=False)
        for _ in range(25):
            view = finder(clean)
        self.assertTrue(finder.ready)
        self.assertTrue(view.found)
        baseline_right = view.model[3]

        bimodal = self._synthetic_bimodal_frame(glare_bimodal=True)
        view_bimodal = finder(bimodal)
        self.assertTrue(view_bimodal.found)
        # Must lock onto true yellow lane line (~2.6), not average with outer grass edge (~5.8).
        self.assertLess(abs(view_bimodal.model[3] - baseline_right), 0.5)

        offroad = np.full((160, 320, 3), (65, 130, 55), dtype=np.uint8)
        for _ in range(5):
            self.assertFalse(finder(offroad).found)


if __name__ == "__main__":
    unittest.main()
