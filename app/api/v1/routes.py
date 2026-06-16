import json
import logging
import tempfile
from typing import AsyncGenerator
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from prometheus_client import Counter
from app.api.v1.schemas import (
    EmotionResponse,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    SentimentRequest,
    SentimentResponse,
    VoiceSentimentResponse,
)
from app.services.rag_service import RAGService
from app.services.memory_service import MemoryService
from app.services.tool_service import ToolService
from app.services.model_service import ModelService
from app.services.writing_service import WritingService
from app.services.sentiment_service import SentimentService
from app.services.emotion_model import EmotionModel
from app.services.voice_pipeline import analyze_voice
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

request_counter = Counter("assistant_requests_total", "Total API requests")
error_counter = Counter("assistant_errors_total", "Total API errors")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", version="1.0.0")


@router.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(payload: SentimentRequest):
    request_counter.inc()
    try:
        return SentimentResponse(**SentimentService.analyze(payload.text))
    except (ValueError, RuntimeError) as exc:
        error_counter.inc()
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        error_counter.inc()
        logger.exception("Sentiment analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/emotion", response_model=EmotionResponse)
async def analyze_emotion(payload: SentimentRequest):
    request_counter.inc()
    try:
        return EmotionResponse(**EmotionModel.analyze(payload.text))
    except (ValueError, RuntimeError) as exc:
        error_counter.inc()
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        error_counter.inc()
        logger.exception("Emotion analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/voice-sentiment", response_model=VoiceSentimentResponse)
async def analyze_voice_sentiment(file: UploadFile = File(...)):
    request_counter.inc()
    suffix = Path(file.filename or "").suffix or ".wav"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
            temp_audio.write(await file.read())
            temp_path = temp_audio.name

        result = analyze_voice(temp_path)
        return VoiceSentimentResponse(**result)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        error_counter.inc()
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        error_counter.inc()
        logger.exception("Voice sentiment analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if "temp_path" in locals():
            Path(temp_path).unlink(missing_ok=True)


@router.post("/infer")
async def infer(payload: InferenceRequest, request: Request):
    request_counter.inc()
    session_id = payload.session_id
    prompt = payload.prompt
    history = payload.history or []

    try:
        await MemoryService.append_user_message(session_id, prompt)

        if payload.use_rag:
            retrieved_docs = await RAGService.retrieve(prompt)
        else:
            retrieved_docs = []

        writing_result = WritingService.maybe_handle(prompt)
        if writing_result:
            await MemoryService.append_assistant_message(session_id, writing_result)
            return InferenceResponse(
                session_id=session_id,
                response=writing_result,
                metadata={"retrieved_docs": len(retrieved_docs), "handled_by": "writing_template"},
            )

        if payload.stream and settings.enable_streaming:
            async def stream_response() -> AsyncGenerator[bytes, None]:
                async for chunk in ModelService.stream_response(
                    session_id=session_id,
                    prompt=prompt,
                    history=history,
                    retrieved_docs=retrieved_docs,
                    tools=payload.tools,
                ):
                    yield chunk

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
            )

        result = await ModelService.generate_response(
            session_id=session_id,
            prompt=prompt,
            history=history,
            retrieved_docs=retrieved_docs,
            tools=payload.tools,
        )

        await MemoryService.append_assistant_message(session_id, result)
        return InferenceResponse(
            session_id=session_id,
            response=result,
            metadata={"retrieved_docs": len(retrieved_docs)},
        )
    except Exception as exc:
        error_counter.inc()
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/tool-callback")
async def tool_callback(payload: dict):
    tool_name = payload.get("tool_name")
    args = payload.get("args", {})
    result = await ToolService.execute(tool_name, args)
    return {"status": "ok", "result": result}
