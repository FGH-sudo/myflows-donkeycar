FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt /app/requirements-deploy.txt
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY apps /app/apps
COPY tools /app/tools
COPY generated /app/generated
COPY MyFlows /app/MyFlows
COPY proto /app/proto

ENV SERVE_MODE=grpc
ENV MODEL_PATH=/models/model.onnx
ENV PORT=50051

COPY deploy/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 50051 8000

ENTRYPOINT ["/entrypoint.sh"]
