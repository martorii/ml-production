FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ARTIFACTS_DIR=/app/artifacts

WORKDIR /app

# Dependencies first so code changes don't invalidate the (slow) install layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tickets/ ./tickets/

# The model is baked into the image at build time. That makes the image a single
# immutable, reproducible unit: this tag == this model. See README for the
# trade-off against loading from a model registry at startup.
RUN python -m tickets.train

RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/ready')"

CMD ["uvicorn", "tickets.api:app", "--host", "0.0.0.0", "--port", "8000"]
