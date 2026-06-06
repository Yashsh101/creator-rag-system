import logging
import os
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.routes import router
from app.api.workflows import router as workflow_router
from app.core.config import settings
from app.core.errors import AppError, app_error_handler, unhandled_error_handler
from app.core.logging import configure_logging
from app.core.rate_limit import rate_limiter
from app.ingest.router import router as ingest_router

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version="0.1.0")


@app.get("/")
async def root():
    return {"status": "ok"}


# Render free tier health-check pings both / and a configurable path.
# Keep a dedicated /health alias so render.yaml healthCheckPath=/health also works.
@app.get("/health")
async def health_root():
    return {"status": "ok"}


@app.get("/api/v1/health")
async def health():
    return {"status": "ok"}


# Build CORS origin list:
#   1. settings.cors_allowed_origins  (comma-separated env var)
#   2. FRONTEND_URL env var           (explicit Vercel URL)
#   3. localhost fallback for local dev
# If nothing is set (e.g. first cold-boot before env vars are configured),
# fall back to ["*"] so the service is reachable at all.
_origins_from_settings = [
    o.strip()
    for o in settings.cors_allowed_origins.split(",")
    if o.strip()
]
_extra_origins = [
    o for o in [
        os.getenv("FRONTEND_URL", ""),
        "http://localhost:3000",
    ]
    if o
]
allowed_origins = list(dict.fromkeys(_origins_from_settings + _extra_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["X-API-Key", "X-Trace-ID", "Content-Type"],
)

app.include_router(router, prefix="/api/v1")
app.include_router(ingest_router, prefix="/api/v1")
app.include_router(workflow_router, prefix="/api/v1")
app.include_router(chat_router)

app.add_exception_handler(AppError, app_error_handler)
app.add_exception_handler(Exception, unhandled_error_handler)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    trace_id = request.headers.get("x-trace-id", uuid.uuid4().hex)
    request.state.trace_id = trace_id
    started = time.perf_counter()

    try:
        # Skip rate-limiting for preflight OPTIONS and health/root paths.
        # Before this fix: OPTIONS preflight was hitting the rate-limiter and
        # returning 429, which caused every CORS request to fail on first touch.
        if (
            request.method != "OPTIONS"
            and request.url.path not in {
                "/",
                "/health",
                "/api/v1/health",
                "/api/v1/health/ready",
            }
        ):
            client_key = (
                request.headers.get("X-API-Key")
                or (request.client.host if request.client else "unknown")
            )
            rate_limiter.check(client_key)

        response = await call_next(request)

    except AppError as exc:
        logger.warning(
            "request_rejected",
            extra={
                "event": "request_rejected",
                "trace_id": trace_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": exc.status_code,
            },
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "trace_id": trace_id,
                }
            },
            headers={"x-trace-id": trace_id},
        )

    except Exception:
        logger.exception(
            "request_failed",
            extra={
                "trace_id": trace_id,
                "method": request.method,
                "path": request.url.path,
            },
        )
        raise

    response.headers["x-trace-id"] = trace_id

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "request_completed method=%s path=%s status=%s latency_ms=%s trace_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        trace_id,
        extra={
            "event": "request_completed",
            "trace_id": trace_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": elapsed_ms,
        },
    )

    return response
