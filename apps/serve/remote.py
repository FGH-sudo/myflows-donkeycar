"""A deployed inference service as a steering function, with the call shape of a local predictor.

    predict = remote_predictor("http://127.0.0.1:8000")    # apps.serve.serve_fastapi
    predict = remote_predictor("grpc://127.0.0.1:50051")   # apps.serve.serve_grpc
    steering = predict(rgb_frame)

A closed-loop run through one of these drives the simulator with the model as it is deployed: the frame leaves
the driving process, the service resizes and normalises it, and the answer comes back over the network stack.
"""

from __future__ import annotations

from urllib.parse import urlsplit

import numpy as np


def remote_predictor(endpoint: str, *, timeout: float = 5.):
    parts = urlsplit(str(endpoint))
    if parts.scheme in ("http", "https") and parts.hostname:
        import cv2

        from .fastapi_client import FastApiInferenceClient

        client = FastApiInferenceClient(str(endpoint), timeout=timeout)

        def predict(rgb) -> float:
            # PNG keeps the frame bit for bit; a second JPEG pass would hand the service a different picture.
            ok, encoded = cv2.imencode(".png", cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError("PNG encoding failed")
            result = client.predict_bytes(encoded.tobytes(), filename="frame.png", content_type="image/png")
            if result.angle is None:
                raise RuntimeError("the inference service returned no steering angle")
            return float(result.angle)

        return predict
    if parts.scheme == "grpc" and parts.hostname and parts.port:
        from .grpc_client import GrpcInferenceClient

        client = GrpcInferenceClient(parts.hostname, parts.port, timeout=timeout)

        def predict(rgb) -> float:
            result = client.predict_array(np.ascontiguousarray(rgb))
            if result.angle is None:
                raise RuntimeError("the inference service returned no steering angle")
            return float(result.angle)

        return predict
    raise ValueError(f"推理服务地址应为 http://host:port 或 grpc://host:port：{endpoint}")
