"""FastAPI service.

  GET  /health   liveness + which model version is loaded
  GET  /ready    readiness: 200 only once the model can actually serve
  POST /predict  classify one ticket
  GET  /metrics  Prometheus exposition (service + drift metrics)
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from tickets import model as model_module
from tickets import monitoring
from tickets.schemas import HealthResponse, PredictionRequest, PredictionResponse
from tickets.text import normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")

STATE: dict = {"model": None, "drift": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup, never inside a request: cold-loading a vectorizer per
    # request is the classic way to destroy p99 latency.
    loaded = model_module.load()
    STATE["model"] = loaded
    STATE["drift"] = monitoring.load_drift_monitor(loaded)
    monitoring.MODEL_INFO.labels(loaded.version).set(1)
    macro_f1 = loaded.metrics.get("metrics", {}).get("macro_f1")
    if macro_f1 is not None:
        monitoring.MODEL_MACRO_F1.set(macro_f1)
    log.info("loaded model_version=%s classes=%s", loaded.version, loaded.classes)
    yield
    STATE.clear()


app = FastAPI(
    title="Support ticket routing",
    version="2.0.0",
    description="Text classification, deployed and monitored like production software.",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Count rejected inputs.

    Without this, schema violations are invisible: pydantic rejects them before any
    handler runs, so the service looks perfectly healthy while callers send garbage.
    Garbage-in rate is a monitoring signal in its own right.
    """
    detail = []
    for error in exc.errors():
        field = ".".join(str(part) for part in error["loc"][1:]) or "body"
        monitoring.REJECTED_REQUESTS.labels(field).inc()
        # Hand-build the response instead of echoing exc.errors(). A custom validator
        # puts its ValueError into `ctx`, which is not JSON serialisable, and the raw
        # errors also echo the caller's input straight back. Type, field and message
        # is everything a client needs.
        detail.append({"field": field, "type": error["type"], "message": error["msg"]})
    monitoring.PREDICTION_ERRORS.labels("validation_error").inc()
    log.warning("rejected request: %s", [e["message"] for e in detail])
    return JSONResponse(status_code=422, content={"detail": detail})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = STATE.get("model")
    return HealthResponse(
        status="ok",
        model_version=loaded.version if loaded else "none",
        model_loaded=loaded is not None,
    )


@app.get("/ready")
def ready() -> dict:
    if STATE.get("model") is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    loaded = STATE.get("model")
    if loaded is None:
        monitoring.PREDICTION_ERRORS.labels("model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="model not loaded")

    started = time.perf_counter()
    try:
        prediction = loaded.predict([request.text])[0]
        embedding = loaded.embed([request.text])[0]
    except Exception:
        monitoring.PREDICTION_ERRORS.labels("inference_error").inc()
        log.exception("inference failed")
        raise HTTPException(status_code=500, detail="inference error") from None
    monitoring.PREDICTION_LATENCY.observe(time.perf_counter() - started)

    monitoring.PREDICTIONS_TOTAL.labels(loaded.version, prediction.label).inc()

    drift = STATE.get("drift")
    if drift is not None:
        # The raw text is never stored or logged; only the normalised form feeds
        # token statistics, and only the embedding is retained in the window.
        drift.observe(normalize(request.text), embedding, prediction.label, prediction.confidence)

    return PredictionResponse(
        label=prediction.label,
        confidence=round(prediction.confidence, 4),
        probabilities={k: round(v, 4) for k, v in prediction.probabilities.items()},
        model_version=loaded.version,
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
