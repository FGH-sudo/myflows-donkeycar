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
            fake_ms.ResNet18 = Mock(return_value=fake_model)
            fake_ms.Graph = Mock(return_value=fake_graph)
            fake_ms.load_checkpoint = Mock()
            fake_ms.export_onnx = Mock(return_value=str(output))

            with (
                patch.object(export_tool, "ms", fake_ms),
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


if __name__ == "__main__":
    unittest.main()
