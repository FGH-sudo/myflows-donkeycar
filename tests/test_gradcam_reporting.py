import unittest
from pathlib import Path

from tools.explain.reporting import append_gradcam_row, gradcam_report_header


class GradcamReportingTest(unittest.TestCase):
    def test_report_header_records_dataset_scope_and_score_meaning(self):
        lines = gradcam_report_header(
            "resnet",
            Path("mycar/models/myflow_resnet18_best"),
            "layer4",
            Path("mycar/logs/tensorboard/gradcam_resnet_test"),
            split="test",
            split_file=Path("mycar/logs/resnet18_split.json"),
            fixed_throttle=0.2,
            force_fixed_throttle=True,
            sample_count=8,
        )

        body = "\n".join(lines)
        self.assertIn("- split: `test` via `mycar/logs/resnet18_split.json`", body)
        self.assertIn("- fixed_throttle: `0.2` forced", body)
        self.assertIn("- samples: `8`", body)
        self.assertIn("score 是 target_output 对应的模型原始预测值", body)
        self.assertIn("| # | image | target_output | score | true_angle | pred_angle | abs_error | overlay |", body)

    def test_report_row_includes_prediction_context(self):
        lines = ["header"]
        append_gradcam_row(
            lines,
            step=0,
            rel_path=Path("images/1000_-0.2222.jpg"),
            target_label="angle",
            score=-0.189347,
            true_angle=-0.2222,
            pred_angle=-0.189347,
            abs_error=0.032853,
            overlay_path=Path("docs/experiments/explainability/run/000_overlay.png"),
            root=Path("."),
        )

        self.assertEqual(
            lines[-1],
            "| 0 | `images/1000_-0.2222.jpg` | `angle` | -0.189347 | -0.222200 | -0.189347 | 0.032853 | `docs/experiments/explainability/run/000_overlay.png` |",
        )


if __name__ == "__main__":
    unittest.main()
