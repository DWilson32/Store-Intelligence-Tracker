"""
main.py — FastAPI entrypoint for Store Intelligence API

Endpoints:
  POST /events/ingest
  GET  /stores/{store_id}/metrics
  GET  /stores/{store_id}/funnel
  GET  /stores/{store_id}/heatmap
  GET  /stores/{store_id}/anomalies
  GET  /health
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.database import init_db, get_db_status
from app.ingestion import router as ingest_router
from app.metrics import router as metrics_router
from app.funnel import router as funnel_router
from app.heatmap import router as heatmap_router
from app.anomalies import router as anomalies_router
from app.health import router as health_router
from app.pos import router as pos_router

# ---------------------------------------------------------------------------
# Structured logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
)
logger = logging.getLogger("store_intelligence")


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('{"event":"startup","message":"Initialising database"}')
    await init_db()
    logger.info('{"event":"startup","message":"Store Intelligence API ready"}')
    yield
    logger.info('{"event":"shutdown","message":"Shutting down"}')


app = FastAPI(
    title="Store Intelligence API",
    description="Real-time retail analytics from CCTV event streams",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request logging middleware — structured logs with trace_id
# ---------------------------------------------------------------------------

@app.middleware("http")
async def structured_logging_middleware(request: Request, call_next):
    trace_id = request.headers.get("X-Trace-Id", str(uuid.uuid4())[:8])
    store_id = request.path_params.get("store_id", "-")
    start = time.perf_counter()

    request.state.trace_id = trace_id

    try:
        response = await call_next(request)
    except Exception as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        logger.error(
            f'{{"trace_id":"{trace_id}","store_id":"{store_id}",'
            f'"endpoint":"{request.url.path}","method":"{request.method}",'
            f'"latency_ms":{latency_ms},"status_code":500,"error":"{str(exc)}"}}'
        )
        return JSONResponse(
            status_code=500,
            content={"error": "internal_server_error", "trace_id": trace_id},
        )

    latency_ms = int((time.perf_counter() - start) * 1000)
    event_count = getattr(request.state, "event_count", None)

    log_parts = (
        f'{{"trace_id":"{trace_id}","store_id":"{store_id}",'
        f'"endpoint":"{request.url.path}","method":"{request.method}",'
        f'"latency_ms":{latency_ms},"status_code":{response.status_code}'
    )
    if event_count is not None:
        log_parts += f',"event_count":{event_count}'
    log_parts += "}"
    logger.info(log_parts)

    response.headers["X-Trace-Id"] = trace_id
    return response


# ---------------------------------------------------------------------------
# Graceful degradation — 503 on DB issues
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    trace_id = getattr(request.state, "trace_id", "unknown")
    logger.error(f"Unhandled exception trace_id={trace_id}: {exc}", exc_info=True)

    # Sanitise — never leak stack traces
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred",
            "trace_id": trace_id,
        },
    )


# ---------------------------------------------------------------------------
# Register routers
# ---------------------------------------------------------------------------

app.include_router(ingest_router)
app.include_router(metrics_router)
app.include_router(funnel_router)
app.include_router(heatmap_router)
app.include_router(anomalies_router)
app.include_router(health_router)
app.include_router(pos_router)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    return {"service": "store-intelligence-api", "version": "1.0.0", "docs": "/docs"}
