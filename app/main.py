import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app
from app.config import settings
from app.logger_config import configure_logging
from app.api.v1.routes import router as api_router
from app.api.v1 import sentiment
from app.db.redis_client import redis_client
from app.services.model_service import ModelService
from app.services.rag_service import RAGService
from app.services.sentiment_service import SentimentService


configure_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Enterprise AI Assistant backend")
    try:
        await redis_client.connect()
    except Exception as e:
        logger.warning("Redis connection failed: %s. Continuing without Redis.", str(e))
    
    try:
        await ModelService.initialize()
    except Exception as e:
        logger.warning("Model initialization failed: %s. API will still start.", str(e))
    
    try:
        RAGService.initialize()
    except Exception as e:
        logger.warning("RAG service initialization failed: %s", str(e))

    try:
        SentimentService.initialize()
    except Exception as e:
        logger.warning("Sentiment service initialization failed: %s", str(e))
        
    yield
    
    logger.info("Shutting down backend")
    await redis_client.disconnect()

app = FastAPI(
    title=settings.app_name,
    description="Enterprise-grade AI assistant backend with RAG, memory and tool calling.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")
app.include_router(sentiment.router, prefix="/api/v1")
app.mount("/metrics", make_asgi_app())
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def chat_app():
    return FileResponse("app/static/index.html")


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Enterprise AI Assistant backend")
    try:
        await redis_client.connect()
    except Exception as e:
        logger.warning("Redis connection failed: %s. Continuing without Redis.", str(e))
    
    try:
        await ModelService.initialize()
    except Exception as e:
        logger.warning("Model initialization failed: %s. API will still start.", str(e))
    
    try:
        RAGService.initialize()
    except Exception as e:
        logger.warning("RAG service initialization failed: %s", str(e))

    try:
        SentimentService.initialize()
    except Exception as e:
        logger.warning("Sentiment service initialization failed: %s", str(e))


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down backend")
    await redis_client.disconnect()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
