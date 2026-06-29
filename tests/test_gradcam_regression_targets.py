import unittest

import numpy as np

from MyFlows.core.graph import Graph
from tools.explain.model_factory import build_gradcam_model
from tools.explain.targets import select_target


class GradcamRegressionTargetsTest(unittest.TestCase):
    def test_model_factory_uses_output_dim_for_resnet_and_vgg(self):
        for model_type in ("resnet", "vgg"):
            _x, _model, outputs = build_gradcam_model(
                model_type,
                h=32,
                w=32,
                output_dim=2,
                dtype=np.float32,
            )
            Graph(outputs).forward()
            self.assertEqual(outputs.value.shape, (1, 2))

    def test_vgg_uses_regression_target_output(self):
        target_index, target_label = select_target(
            "vgg",
            np.asarray([0.2, 0.4], dtype=np.float32),
            target_output="throttle",
        )

        self.assertEqual(target_index, 1)
        self.assertEqual(target_label, "throttle")


if __name__ == "__main__":
    unittest.main()
