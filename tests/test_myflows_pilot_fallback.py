import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_pilot_module():
    path = ROOT / "mycar" / "myflows_pilot.py"
    spec = importlib.util.spec_from_file_location("myflows_pilot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MyFlowsPilotFallbackTest(unittest.TestCase):
    def test_fallback_constructs_resnet18_with_output_dim(self):
        mod = _load_pilot_module()
        fake_model = Mock()
        fake_out = Mock()
        fake_out.value = np.array([[0.25, 0.8]], dtype=np.float32)
        fake_model.return_value = fake_out
        fake_graph = Mock()
        fake_graph.forward = Mock()

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "best"
            with (
                patch.object(mod, "resolve_myflows_device", return_value="cpu"),
                patch.object(mod.ms, "ResNet18", return_value=fake_model) as ctor,
                patch.object(mod.ms, "Graph", return_value=fake_graph),
                patch.object(mod.ms, "Variable", return_value=Mock()),
                patch.object(mod.ms, "load_checkpoint"),
            ):
                pilot = mod.MyFlowsResNet18Pilot(str(ckpt), image_w=16, image_h=16, device="cpu")

        self.assertIsNone(pilot.session)
        kwargs = ctor.call_args.kwargs
        self.assertEqual(kwargs.get("output_dim"), 2)
        self.assertNotIn("num_classes", kwargs)
        fake_graph.forward.assert_called()

        img = np.zeros((16, 16, 3), dtype=np.uint8)
        angle, throttle = pilot.run(img)
        self.assertTrue(np.isfinite(angle))
        self.assertEqual(throttle, pilot.fixed_throttle)


if __name__ == "__main__":
    unittest.main()
