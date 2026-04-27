from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes import router
from app.core.logging import setup_logging, logger
from app.core.config import settings

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FairSight Core service starting up...")
    yield
    logger.info("FairSight Core service shutting down.")


app = FastAPI(
    title="FairSight Core",
    description="Enterprise-grade AI fairness analysis and bias mitigation platform.",
    version="1.1.0",
    lifespan=lifespan,
)

# CORS — no credentials needed for this API, so wildcard origin is safe
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "service": "FairSight Core", "version": "1.1.0"}
