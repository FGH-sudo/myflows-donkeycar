import unittest

from benchmark.serve_bench import build_bench_stats


class ServeBenchDeviceFieldsTest(unittest.TestCase):
    def test_local_mode_reports_actual_in_process_device(self):
        stats = build_bench_stats(
            mode="local",
            workers=4,
            model="mycar/models/myflow_resnet18_best.onnx",
            device="cuda",
            latencies=[1.0, 2.0],
            errors=0,
            requested=2,
        )

        self.assertEqual(stats["device"], "cuda")
        self.assertNotIn("server_device", stats)

    def test_remote_modes_do_not_report_client_default_as_device(self):
        for mode in ("grpc", "fastapi"):
            stats = build_bench_stats(
                mode=mode,
                workers=4,
                model="mycar/models/myflow_resnet18_best.onnx",
                device="cpu",
                latencies=[1.0, 2.0],
                errors=0,
                requested=2,
            )

            self.assertNotIn("device", stats)
            self.assertEqual(stats["client_device"], "n/a")
            self.assertEqual(stats["server_device"], "controlled_by_service")


if __name__ == "__main__":
    unittest.main()
