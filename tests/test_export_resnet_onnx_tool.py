import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import tools.export_resnet_onnx as export_tool


class ExportResNetOnnxToolTest(unittest.TestCase):
    def test_export_resnet18_onnx_uses_requested_batch_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "best"
            checkpoint.with_suffix(".json").write_text("{}", encoding="utf-8")
            np.savez(checkpoint.with_suffix(".npz"), dummy=np.array([1], dtype=np.float32))
            output = Path(tmp) / "best_b1.onnx"

            fake_ms = SimpleNamespace()
            fake_x = object()
            fake_logits = object()
            fake_model = Mock(return_value=fake_logits)
            fake_graph = Mock()
            fake_graph.forward = Mock()
            fake_ms.Variable = Mock(return_value=fake_x)
            fake_ms.Graph = Mock(return_value=fake_graph)
            fake_ms.load_checkpoint = Mock()
            fake_ms.export_onnx = Mock(return_value=str(output))

            with (
                patch.object(export_tool, "ms", fake_ms),
                patch.object(export_tool, "build_regression_resnet", return_value=fake_model) as build_model,
                patch.object(export_tool, "resolve_myflows_device", return_value="cpu"),
                patch.object(export_tool, "print_myflows_device"),
            ):
                exported = export_tool.export_resnet18_onnx(
                    checkpoint,
                    output=output,
                    batch_size=1,
                    image_w=160,
                    image_h=120,
                    device="cpu",
                )

            self.assertEqual(exported, output)
            self.assertEqual(build_model.call_args.args, ("resnet18", "global", 120, 160))
            fake_ms.Variable.assert_called_once()
            x_value = fake_ms.Variable.call_args.args[0]
            self.assertEqual(x_value.shape, (1, 3, 120, 160))
            fake_ms.load_checkpoint.assert_called_once_with([fake_model], None, str(checkpoint))
            fake_ms.export_onnx.assert_called_once_with(
                fake_graph,
                str(output),
                input_nodes=[fake_x],
                output_names=["control"],
            )

    def test_export_vgg11_onnx_uses_regression_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "vgg_best"
            checkpoint.with_suffix(".json").write_text("{}", encoding="utf-8")
            np.savez(checkpoint.with_suffix(".npz"), dummy=np.array([1], dtype=np.float32))
            output = Path(tmp) / "vgg_best.onnx"

            fake_ms = SimpleNamespace()
            fake_x = object()
            fake_control = object()
            fake_model = Mock(return_value=fake_control)
            fake_graph = Mock()
            fake_graph.forward = Mock()
            fake_ms.Variable = Mock(return_value=fake_x)
            fake_ms.VGG11 = Mock(return_value=fake_model)
            fake_ms.Graph = Mock(return_value=fake_graph)
            fake_ms.load_checkpoint = Mock()
            fake_ms.export_onnx = Mock(return_value=str(output))

            with (
                patch.object(export_tool, "ms", fake_ms),
                patch.object(export_tool, "resolve_myflows_device", return_value="cpu"),
                patch.object(export_tool, "print_myflows_device"),
            ):
                exported = export_tool.export_vgg11_onnx(
                    checkpoint,
                    output=output,
                    batch_size=1,
                    image_w=160,
                    image_h=120,
                    device="cpu",
                )

            self.assertEqual(exported, output)
            x_value = fake_ms.Variable.call_args.args[0]
            self.assertEqual(x_value.shape, (1, 3, 120, 160))
            fake_ms.VGG11.assert_called_once_with(
                in_channels=3,
                output_dim=2,
                image_h=120,
                image_w=160,
                name="vgg11_donkey",
            )
            fake_ms.load_checkpoint.assert_called_once_with([fake_model], None, str(checkpoint))
            fake_ms.export_onnx.assert_called_once_with(
                fake_graph,
                str(output),
                input_nodes=[fake_x],
                output_names=["control"],
            )

    def test_export_resnet_onnx_dynamic_batch_works_with_predictor_and_batched_eval(self):
        import json
        import subprocess
        import sys
        import cv2
        import MyFlows as ms
        from MyFlows.core.device import set_device
        from MyFlows.core.dtype import set_default_dtype
        from apps.serve.onnx_predictor import OnnxPredictor
        from apps.train.common.regression_model import build_regression_resnet

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            images = work / "tub" / "images"
            images.mkdir(parents=True)
            rng = np.random.default_rng(19)
            for idx in range(6):
                rgb = rng.integers(0, 255, size=(64, 96, 3), dtype=np.uint8)
                angle = -0.5 + 0.2 * idx
                cv2.imwrite(str(images / f"{idx}_{angle:.4f}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            ckpt = work / "model"
            try:
                set_device("cpu")
                set_default_dtype(np.float32)
                x = ms.Variable(np.zeros((1, 3, 48, 80), dtype=np.float32), name="X", requires_grad=False)
                model = build_regression_resnet(
                    "resnet18", "global", 48, 80, in_channels=3, output_dim=2, stem="cifar",
                    base_width=4, name="resnet18_donkey",
                )
                model.eval()
                ms.Graph(model(x)).forward()
                ms.save_checkpoint([model], None, 0, filepath=str(ckpt), loss=0.1)
            finally:
                set_device("cpu")
                set_default_dtype(np.float64)

            onnx_path = export_tool.export_resnet_onnx(
                ckpt, batch_size=1, image_w=80, image_h=48, base_width=4,
                device="cpu", roi_crop="16,8", dynamic_batch=True,
            )
            self.assertTrue(onnx_path.name.endswith("_dynamic.onnx"))

            predictor = OnnxPredictor(onnx_path, 80, 48, device="cpu")
            self.assertEqual(predictor.roi_crop, (16, 8, 0, 0))
            sample_rgb = cv2.cvtColor(cv2.imread(str(images / "0_-0.5000.jpg")), cv2.COLOR_BGR2RGB)
            pred = predictor.predict(sample_rgb)
            self.assertEqual(pred.shape, (2,))

            report_json = work / "eval_b4.json"
            proc = subprocess.run(
                [
                    sys.executable, "-X", "utf8", "-m", "apps.eval.eval_myflows_donkey_onnx",
                    "--data", str(work / "tub"), "--catalog", "missing.catalog",
                    "--checkpoint", str(onnx_path), "--batch", "4", "--device", "cpu",
                    "--report-json", str(report_json),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            report = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertEqual(report["results"][0]["batch"], 4)
            self.assertEqual(report["results"][0]["roi_crop"], "16,8")


if __name__ == "__main__":
    unittest.main()
