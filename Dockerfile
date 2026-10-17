# The service itself (FastAPI, queue, rules, dashboard). The model runs in a separate container (see docker-compose.yml)
# and is reached over the private compose network; video frames never leave the machine.
FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart av pillow numpy pyyaml httpx
COPY app ./app

ENV VLM_BACKEND=openai \
    VLM_URL=http://llama:8081 \
    VLM_ALLOWED_HOSTS=llama \
    DATA_DIR=/data \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=4s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning"]
