# app/main.py
from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────
import asyncio
import contextlib
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

# ── third-party ───────────────────────────────────────────────────────────
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

# ── local imports ────────────────────────────────────────────────────────
from .cache import default_cache
from .models import (
    APIResponse,
    Article,
    ErrorResponse,
    HealthCheck,
    ReachCluster,
    SentimentSummary,
)
from .news_fetcher import get_articles, periodic_feed_update
from .reach import cluster_with_labels, get_or_load_similarity_model
from .sentiment import get_or_load_sentiment_model, sentiment_counts

# ──────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("news-analytics")

# ──────────────────────────────────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────────────────────────────────
ml_models: dict[str, object] = {}
background_tasks_manager: Optional[asyncio.Task] = None

# ──────────────────────────────────────────────────────────────────────────
# Middleware
# ──────────────────────────────────────────────────────────────────────────
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        logger.info("► %s %s", request.method, request.url.path)
        try:
            response = await call_next(request)
        except Exception:
            # re-raise *after* logging duration
            raise
        finally:
            duration = time.time() - start
            logger.info("◄ %s %.3fs %s", request.method, duration, request.url.path)
            if "response" in locals():
                response.headers["X-Process-Time"] = f"{duration:.3f}"
        return response


# ──────────────────────────────────────────────────────────────────────────
# Lifespan
# ──────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown orchestration."""
    logger.info("⇢ Startup")
    try:
        # ① load ML models
        logger.info("Loading ML models …")
        ml_models["sentiment"] = await get_or_load_sentiment_model()
        ml_models["similarity"] = await get_or_load_similarity_model()
        logger.info("✓ ML models ready")

        # ② kick off background cron (hourly refresh)
        global background_tasks_manager
        background_tasks_manager = asyncio.create_task(periodic_feed_update(3600))
        logger.info("✓ Background task scheduled")

        # ③ warm the cache once
        await get_articles()
        logger.info("✓ Initial RSS ingest complete")

        yield  # ── app runs ───────────────────────────────────────────────

    finally:
        logger.info("⇠ Shutdown")
        if background_tasks_manager:
            background_tasks_manager.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await background_tasks_manager
        ml_models.clear()
        await default_cache.clear()
        logger.info("✓ Shutdown complete")


# ──────────────────────────────────────────────────────────────────────────
# FastAPI instance + core middleware
# ──────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="News-Analytics API",
    version="2.0.0",
    description="Advanced RSS aggregation, sentiment & reach analytics.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# ──────────────────────────────────────────────────────────────────────────
# Debug router – lets you inspect/clear the article cache ❶
# ──────────────────────────────────────────────────────────────────────────
debug = APIRouter(prefix="/__debug", tags=["debug"])


@debug.get("/cache/articles")
async def debug_view_cache(limit: int = Query(5, ge=1, le=1000)):
    """
    Inspect what is stored under the *articles* key in `default_cache`.
    """
    raw = await default_cache.get("articles") or []
    return {
        "cached_count": len(raw),
        "sample": raw[:limit],
    }


@debug.delete("/cache/articles")
async def debug_clear_cache():
    """Erase the cached article list manually."""
    await default_cache.delete("articles")
    return {"message": "cache cleared"}


app.include_router(debug)

# ──────────────────────────────────────────────────────────────────────────
# Helper dependency
# ──────────────────────────────────────────────────────────────────────────
async def _articles_dep() -> List[Article]:
    art = await get_articles()
    if not art:
        raise HTTPException(503, "No articles available yet")
    return art


# ──────────────────────────────────────────────────────────────────────────
# Public endpoints  (all logic from your previous file, unchanged)
# ──────────────────────────────────────────────────────────────────────────
@app.get("/", tags=["status"])
async def root():
    return {
        "name": "News-Analytics API",
        "version": "2.0.0",
        "status": "healthy",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthCheck, tags=["status"])
async def health():
    cache_ok = False
    try:
        await default_cache.set("ping", "pong", ttl=30)
        cache_ok = (await default_cache.get("ping")) == "pong"
    except Exception:
        cache_ok = False

    return HealthCheck(
        status="healthy" if cache_ok else "degraded",
        checks={
            "cache": "ok" if cache_ok else "error",
            "sentiment_model": "loaded" if "sentiment" in ml_models else "missing",
            "similarity_model": "loaded" if "similarity" in ml_models else "missing",
        },
    )


@app.get("/articles", response_model=List[Article], tags=["content"])
async def list_articles(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    source: Optional[str] = Query(None),
    articles: List[Article] = Depends(_articles_dep),
):
    if source:
        articles = [a for a in articles if source.lower() in a.source.lower()]
    return articles[offset : offset + limit]


@app.get("/analytics/sentiment", response_model=SentimentSummary, tags=["analytics"])
async def sentiment_endpoint(articles: List[Article] = Depends(_articles_dep)):
    return await sentiment_counts(articles, ml_models["sentiment"])


@app.get("/analytics/reach", response_model=List[ReachCluster], tags=["analytics"])
async def reach_endpoint(
    threshold: float = Query(0.6, ge=0.1, le=1.0),
    top: int = Query(20, ge=1, le=100),
    articles: List[Article] = Depends(_articles_dep),
):
    clusters, _ = await cluster_with_labels(articles, distance_threshold=threshold)
    return clusters[:top]


@app.get("/analytics/trend", tags=["analytics"])
async def trend_endpoint(articles: List[Article] = Depends(_articles_dep)):
    summary = await sentiment_counts(articles, ml_models["sentiment"])
    if summary.total == 0:
        raise HTTPException(503, "No articles to analyse")

    return {
        "total": summary.total,
        "positive_pct": round(summary.positive_ratio * 100, 2),
        "overall_trend": "positive"
        if summary.positive >= summary.negative
        else "negative",
        "generated_at": datetime.utcnow().isoformat(),
    }


@app.post("/admin/refresh", tags=["admin"])
async def admin_refresh(background: BackgroundTasks):
    background.add_task(get_articles, force_refresh=True)
    return {"message": "refresh queued", "timestamp": datetime.utcnow().isoformat()}


# ──────────────────────────────────────────────────────────────────────────
# Global exception handlers (kept concise)
# ──────────────────────────────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def _val_err(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="validation_error",
            message=str(exc),
            details=exc.errors(),
        ).model_dump(),
    )


@app.exception_handler(StarletteHTTPException)
async def _http_err(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error="http_error", message=exc.detail).model_dump(),
    )


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    logger.error("Unhandled exception", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="server_error", message=str(exc)).model_dump(),
    )


# ──────────────────────────────────────────────────────────────────────────
#  dev entry-point
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
