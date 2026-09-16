FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ARTIFACTS_DIR=/app/artifacts

WORKDIR /app

# Dependencies first so code changes don't invalidate the (slow) install layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tickets/ ./tickets/

# The model is baked into the image, but it is NOT trained here: it is copied in from
# the CI job that trained it and ran the quality gate against it. Training during the
# build would produce a second, different model — the gate would then be validating an
# artifact that never ships. Copying makes the tested model the shipped model, and keeps
# the image a single immutable unit: this tag == this model. See README for the
# trade-off against loading from a model registry at startup.
#
# Build locally with `make image` (or `make train` first); a bare `docker build` with no
# artifacts/ present fails the check below rather than silently shipping no model.
COPY artifacts/ ./artifacts/

RUN test -f artifacts/model.joblib && test -f artifacts/metrics.json \
    && test -f artifacts/drift_reference.json \
    || (echo "ERROR: artifacts/ incomplete — run 'python -m tickets.train' before building" >&2; exit 1)

RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/ready')"

CMD ["uvicorn", "tickets.api:app", "--host", "0.0.0.0", "--port", "8000"]
