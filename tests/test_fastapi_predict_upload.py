import base64
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from apps.serve import config as serve_config
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
    def test_json_prediction_and_openapi_resolve_the_request_body(self):
        ok, encoded = cv2.imencode(".png", np.full((12, 16, 3), 127, dtype=np.uint8))
        self.assertTrue(ok)
        with tempfile.TemporaryDirectory() as tmp, patch.object(serve_fastapi, "OnnxPredictor", _FakePredictor):
            client = TestClient(serve_fastapi.create_app(Path("fake.onnx"), log_file=Path(tmp) / "serve.jsonl"))
            response = client.post("/predict_json", json={"image_b64": base64.b64encode(encoded).decode(),
                                                         "width": 16, "height": 12})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["outputs"], [0.1, 0.2])
            schema = client.get("/openapi.json")
            self.assertEqual(schema.status_code, 200, schema.text)
            self.assertIn("requestBody", schema.json()["paths"]["/predict_json"]["post"])

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

    def client(self, **kwargs):
        with patch.object(serve_fastapi, "OnnxPredictor", _FakePredictor):
            return TestClient(serve_fastapi.create_app(Path("fake.onnx"), log_file=None, **kwargs),
                              raise_server_exceptions=False)

    def test_request_bodies_are_bounded_before_they_are_parsed(self):
        limit_mb = 0.25
        client = self.client(max_upload_mb=limit_mb)
        too_big = b"\xff" * int(1.5 * 1024 * 1024)
        with patch.object(serve_fastapi.cv2, "imdecode") as decode:
            upload = client.post("/predict", files={"file": ("big.jpg", too_big, "image/jpeg")})
            self.assertEqual(upload.status_code, 413, upload.text)
            as_json = client.post("/predict_json", json={"image_b64": base64.b64encode(too_big).decode()})
            self.assertEqual(as_json.status_code, 413, as_json.text)

            def chunks():  # no Content-Length: the limit has to hold while the body streams in
                for _ in range(24):
                    yield b"x" * 65536

            streamed = client.post("/predict_json", content=chunks(), headers={"content-type": "application/json"})
            self.assertEqual(streamed.status_code, 413, streamed.text)
            self.assertNotIn("content-length", {k.lower() for k in streamed.request.headers})
            for bad in ("-1", "abc"):
                response = client.post("/predict_json", content=b"{}", headers={"content-length": bad})
                self.assertEqual(response.status_code, 400, (bad, response.text))
            decode.assert_not_called()
        # Other routes are not limited, and a body inside the limit still works.
        self.assertEqual(client.get("/healthz").json(), {"status": "ok"})
        ok, encoded = cv2.imencode(".png", np.full((12, 16, 3), 127, dtype=np.uint8))
        fine = client.post("/predict", files={"file": ("ok.png", encoded.tobytes(), "image/png")})
        self.assertEqual(fine.status_code, 200, fine.text)
        for value in (0, -1, float("nan"), float("inf"), 1e-9):
            with self.assertRaises(ValueError, msg=value), patch.object(serve_fastapi, "OnnxPredictor", _FakePredictor):
                serve_fastapi.create_app(Path("fake.onnx"), log_file=None, max_upload_mb=value)

    def test_image_header_is_checked_before_the_bitmap_is_decoded(self):
        def png_header_only(width, height):
            def chunk(kind, data):
                return (struct.pack(">I", len(data)) + kind + data
                        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

            header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
            return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"")) + chunk(b"IEND", b"")

        client = self.client()
        with patch.object(serve_fastapi.cv2, "imdecode") as decode:
            # A few dozen bytes that claim 2.5e9 / 3.6e7 pixels: refused from the header alone.
            for size in ((50000, 50000), (6000, 6000)):
                bomb = png_header_only(*size)
                self.assertLess(len(bomb), 200)
                response = client.post("/predict", files={"file": ("bomb.png", bomb, "image/png")})
                self.assertEqual(response.status_code, 413, (size, response.text))
            for junk in (b"not an image", b"GIF89a" + b"\x00" * 32, b"\x89PNG\r\n\x1a\n" + b"\x00" * 16):
                response = client.post("/predict", files={"file": ("junk.png", junk, "image/png")})
                self.assertEqual(response.status_code, 400, (junk, response.text))
            decode.assert_not_called()
        empty = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
        self.assertEqual(empty.status_code, 400, empty.text)
        wrong_type = client.post("/predict", files={"file": ("a.txt", b"x", "text/plain")})
        self.assertEqual(wrong_type.status_code, 400, wrong_type.text)


class ServeBindingTest(unittest.TestCase):
    def test_fixed_onnx_batch_and_resolution_across_http_grpc_and_pilot(self):
        from concurrent.futures import ThreadPoolExecutor

        import onnx
        from onnx import TensorProto, helper

        from apps.serve import serve_grpc
        from apps.serve.onnx_predictor import OnnxPredictor
        from mycar.myflows_pilot import MyFlowsResNet18Pilot

        rgb = np.full((12, 16, 3), (51, 102, 204), dtype=np.uint8)
        ok, png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.assertTrue(ok)
        graph = helper.make_graph(
            [helper.make_node("ReduceMean", ["image"], ["prediction"], axes=[2, 3], keepdims=0)],
            "rgb-means", [helper.make_tensor_value_info("image", TensorProto.FLOAT, [2, 3, 4, 6])],
            [helper.make_tensor_value_info("prediction", TensorProto.FLOAT, [2, 3])],
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "model.onnx"
            onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8), path)
            with TestClient(serve_fastapi.create_app(path, device="cpu", log_file=root / "http.jsonl")) as client:
                info = client.get("/model_info").json()
                self.assertEqual((info["image_w"], info["image_h"]), (6, 4))
                http = client.post("/predict", files={"file": ("frame.png", png.tobytes(), "image/png")})
                self.assertEqual(http.status_code, 200, http.text)
            predictor = OnnxPredictor(path, 160, 120, device="cpu")
            with ThreadPoolExecutor(max_workers=1) as pool:
                grpc = serve_grpc.grpc
                server = grpc.server(pool)
                port = server.add_insecure_port("127.0.0.1:0")
                serve_grpc.infer_pb2_grpc.add_InferServiceServicer_to_server(
                    serve_grpc.InferServicer(predictor, root / "grpc.jsonl"), server)
                server.start()
                try:
                    with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
                        stub = serve_grpc.infer_pb2_grpc.InferServiceStub(channel)
                        out = stub.Predict(serve_grpc.infer_pb2.PredictRequest(
                            image_rgb=rgb.tobytes(), width=rgb.shape[1], height=rgb.shape[0], channels=3), timeout=5)
                finally:
                    server.stop(0).wait(5)
            pilot = MyFlowsResNet18Pilot(path, device="cpu")
            np.testing.assert_allclose(http.json()["outputs"], [.2, .4, .8], atol=1e-6)
            np.testing.assert_array_equal(list(out.outputs), http.json()["outputs"])
            self.assertAlmostEqual(pilot.run(rgb)[0], .2, places=6)

    def test_servers_listen_on_loopback_unless_told_otherwise(self):
        self.assertEqual(serve_config.DEFAULT_BIND_HOST, "127.0.0.1")
        self.assertEqual(serve_config.grpc_bind_address("127.0.0.1", 50051), "127.0.0.1:50051")
        self.assertEqual(serve_config.grpc_bind_address("0.0.0.0", 50051), "[::]:50051")
        self.assertEqual(serve_config.grpc_bind_address("::1", 50051), "[::1]:50051")
        self.assertEqual(serve_config.grpc_bind_address("[::1]", "50051"), "[::1]:50051")
        self.assertEqual(serve_config.grpc_bind_address("infer.local", 1), "infer.local:1")
        for host in ("127.0.0.1", "localhost", "::1", "[::1]", "LOCALHOST"):
            self.assertIsNone(serve_config.exposure_warning(host), host)
        for host in ("0.0.0.0", "::", "192.168.1.20", "infer.local"):
            self.assertIn(host, serve_config.exposure_warning(host))
        entrypoint = (Path(serve_config.ROOT) / "deploy" / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
        launches = [line for line in entrypoint.splitlines() if "apps.serve.serve_" in line]
        self.assertEqual(len(launches), 2)
        self.assertTrue(all("--host 0.0.0.0" in line for line in launches), "the container publishes its port")

    def test_grpc_failures_are_logged_as_errors(self):
        from apps.serve import serve_grpc

        class Context:
            code = details = None

            def set_code(self, code):
                self.code = code

            def set_details(self, details):
                self.details = details

        class Failing(_FakePredictor):
            model_type = "regression"

            def predict(self, rgb):
                raise RuntimeError("provider lost")

        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "grpc.jsonl"
            servicer = serve_grpc.InferServicer(Failing(), log, max_pixels=64)
            request = serve_grpc.infer_pb2.PredictRequest(image_rgb=bytes(4 * 4 * 3), width=4, height=4, channels=3)
            with self.assertRaisesRegex(RuntimeError, "provider lost"):
                servicer.Predict(request, Context())
            context = Context()
            servicer.Predict(serve_grpc.infer_pb2.PredictRequest(image_rgb=bytes(3), width=4, height=4, channels=3),
                             context)
            self.assertEqual(context.code, serve_grpc.grpc.StatusCode.INVALID_ARGUMENT)
            context = Context()
            servicer.Predict(serve_grpc.infer_pb2.PredictRequest(image_rgb=bytes(3), width=9, height=9, channels=3),
                             context)
            self.assertEqual(context.code, serve_grpc.grpc.StatusCode.RESOURCE_EXHAUSTED)
            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["status"] for row in rows], ["error", "error", "error"])
            self.assertIn("RuntimeError: provider lost", rows[0]["error"])
            self.assertEqual(servicer.metrics.snapshot()["total_errors"], 3)

    def test_predict_json_omits_resize_when_dimensions_unspecified_and_rejects_partial_dims(self):
        shapes = []

        class _SpyPredictor(_FakePredictor):
            def predict(self, rgb):
                shapes.append(rgb.shape)
                return super().predict(rgb)

        ok, encoded = cv2.imencode(".png", np.full((30, 40, 3), 120, dtype=np.uint8))
        self.assertTrue(ok)
        b64 = base64.b64encode(encoded).decode()
        with patch.object(serve_fastapi, "OnnxPredictor", _SpyPredictor):
            with TestClient(serve_fastapi.create_app(Path("fake.onnx"), log_file=None)) as client:
                r = client.post("/predict_json", json={"image_b64": b64})
                self.assertEqual(r.status_code, 200, r.text)
                self.assertEqual(shapes[-1], (30, 40, 3))
                self.assertEqual(r.json()["input_shape"], [30, 40, 3])
                self.assertEqual(client.post("/predict_json", json={"image_b64": b64, "width": 20}).status_code, 400)
                self.assertEqual(client.post("/predict_json", json={"image_b64": b64, "height": 20}).status_code, 400)

    def test_grpc_create_server_accepts_payloads_above_default_4mib_limit(self):
        from apps.serve import serve_grpc

        class _SpyPredictor(_FakePredictor):
            model_type = "regression"

            def predict(self, rgb):
                return np.asarray([0.1, 0.2], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            server = serve_grpc.create_server(_SpyPredictor(), Path(tmp) / "grpc.jsonl", workers=1)
            port = server.add_insecure_port("127.0.0.1:0")
            server.start()
            try:
                opts = [("grpc.max_send_message_length", 16 * 1024 * 1024)]
                with serve_grpc.grpc.insecure_channel(f"127.0.0.1:{port}", options=opts) as channel:
                    stub = serve_grpc.infer_pb2_grpc.InferServiceStub(channel)
                    big_rgb = bytes(1920 * 1080 * 3)  # > 6.2 MB, exceeds default 4 MiB gRPC limit
                    res = stub.Predict(serve_grpc.infer_pb2.PredictRequest(
                        image_rgb=big_rgb, width=1920, height=1080, channels=3), timeout=5)
                np.testing.assert_allclose(list(res.outputs), [0.1, 0.2], atol=1e-6)
            finally:
                server.stop(0).wait(5)


if __name__ == "__main__":
    unittest.main()
