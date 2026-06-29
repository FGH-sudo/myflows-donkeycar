from pathlib import Path
import unittest


class VggExportUsesToolTest(unittest.TestCase):
    def test_vgg_training_export_delegates_to_tool_script(self):
        source = Path("apps/train/train_vgg_donkey_regression.py").read_text(encoding="utf-8")
        self.assertIn("from tools.export_resnet_onnx import export_vgg11_onnx", source)
        self.assertIn("export_vgg11_onnx(", source)
        self.assertNotIn("ms.export_onnx(", source)


if __name__ == "__main__":
    unittest.main()
