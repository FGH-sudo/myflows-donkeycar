import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from apps.serve import serve_fastapi


class _FakePredictor:
    device = "cuda"

    def __init__(self, *args, **kwargs):
        self.model_path = args[0] if args else Path("fake.onnx")
        self.image_w = 160
        self.image_h = 120
        self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def info(self):
        return {
            "model_path": str(self.model_path),
            "image_w": self.image_w,
            "image_h": self.image_h,
            "providers": self.providers,
            "device": self.device,
        }

    def predict(self, rgb):
        return np.asarray([[0.1, 0.2]], dtype=np.float32)

    def format_prediction(self, out):
        return {"model_type": "regression", "outputs": [0.1, 0.2], "angle": 0.1, "throttle": 0.2}


class FastApiPredictUploadTest(unittest.TestCase):
    def test_predict_accepts_multipart_upload(self):
        image = np.full((12, 16, 3), 127, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)

        with patch.object(serve_fastapi, "OnnxPredictor", _FakePredictor):
            app = serve_fastapi.create_app(Path("fake.onnx"), log_file=None)
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post(
                "/predict",
                files={"file": ("sample.jpg", encoded.tobytes(), "image/jpeg")},
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["device"], "cuda")
        self.assertEqual(payload["outputs"], [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
