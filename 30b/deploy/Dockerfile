# Python 3.12 / CUDA 13, Linux, single 80GB-class GPU.
# Build: docker build -t juxy30b .
# Run:   docker run --gpus all -p 8000:8000 -e HF_TOKEN=... -e JUXY_API_KEY=... juxy30b
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    JUXY_PORT=8000 \
    JUXY_ADAPTER=zxcxzczxcxzc/juxy-maxx-30b

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Versions pinned to the ones the adapter was trained and evaluated with.
RUN pip install --no-cache-dir \
      torch==2.14.1 \
      transformers==5.14.1 \
      peft==0.21.2 \
      accelerate==1.15.0 \
      fastapi==0.162.0 \
      uvicorn==0.34.0 \
      huggingface_hub==1.33.0 \
      hf_transfer \
      sentencepiece

WORKDIR /app
COPY serve_juxy.py /app/serve_juxy.py

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=900s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["python", "/app/serve_juxy.py"]
