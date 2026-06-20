import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Any

from app.services.emotion_model import EmotionModel
from app.services.nlp_utils import preprocess_text
from app.services.tool_service import ToolService

logger = logging.getLogger(__name__)

# Patterns that suggest the user wants real-time/live information
REALTIME_PATTERNS = [
    re.compile(r"\b(weather|temperature|forecast)\b", re.IGNORECASE),
    re.compile(r"\b(news|headlines|latest|breaking)\b", re.IGNORECASE),
    re.compile(r"\b(stock|price|market|crypto|bitcoin)\b", re.IGNORECASE),
    re.compile(r"\b(score|match|game|sports|live)\b", re.IGNORECASE),
    re.compile(r"\b(what('s| is) happening|what('s| is) going on)\b", re.IGNORECASE),
    re.compile(r"\b(today|now|current|right now|at the moment)\b", re.IGNORECASE),
    re.compile(r"\b(search|find|look up|google|check)\b", re.IGNORECASE),
]


def transcribe_audio(audio_path: str, model_size: str | None = None) -> str:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Audio path is not a file: {path}")

    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "openai-whisper is required for voice-to-text. "
            "Install project requirements first."
        ) from exc

    selected_model = model_size or os.getenv("WHISPER_MODEL_SIZE", "base")
    model = whisper.load_model(selected_model)
    result = model.transcribe(str(path))
    transcript = str(result.get("text", "")).strip()
    if not transcript:
        raise RuntimeError("Whisper returned an empty transcript")
    return transcript


def _needs_realtime_data(transcript: str) -> bool:
    """Check if the transcript suggests the user wants live/real-time information."""
    for pattern in REALTIME_PATTERNS:
        if pattern.search(transcript):
            return True
    return False


def _fetch_realtime_context(transcript: str) -> List[Dict[str, Any]]:
    """Fetch real-time data using available tools based on the transcript."""
    context_items = []

    # Try web search for general real-time queries
    try:
        search_result = ToolService.execute("web_search", {"query": transcript, "max_results": 3})
        if search_result and isinstance(search_result, dict) and "results" in search_result:
            results = search_result.get("results", [])
            if results:
                context_items.append({
                    "source": "web_search",
                    "query": transcript,
                    "results": results,
                })
    except Exception as exc:
        logger.warning("Real-time web search failed: %s", str(exc))

    # Try knowledge base search as fallback
    try:
        kb_result = ToolService.execute("search_knowledge_base", {"query": transcript, "top_k": 3})
        if kb_result and isinstance(kb_result, dict) and "results" in kb_result:
            results = kb_result.get("results", [])
            if results:
                context_items.append({
                    "source": "knowledge_base",
                    "query": transcript,
                    "results": results,
                })
    except Exception as exc:
        logger.warning("Knowledge base search failed: %s", str(exc))

    return context_items


def analyze_voice(audio_path: str, model_size: str | None = None) -> Dict[str, object]:
    transcript = transcribe_audio(audio_path, model_size=model_size)
    processed_text = preprocess_text(transcript)
    sentiment = EmotionModel.analyze(processed_text)

    # Check if we should fetch real-time data
    realtime_context = []
    if _needs_realtime_data(transcript):
        logger.info("Voice query appears to need real-time data: %s", transcript)
        realtime_context = _fetch_realtime_context(transcript)

    result = {
        "audio_path": str(audio_path),
        "transcript": transcript,
        "processed_text": processed_text,
        "sentiment": sentiment["sentiment"],
        "confidence": sentiment["confidence"],
        "probabilities": sentiment["probabilities"],
        "model": sentiment["model"],
        "realtime_data_available": len(realtime_context) > 0,
        "realtime_context": realtime_context,
    }

    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Voice -> Text -> Sentiment")
    parser.add_argument("audio_path", help="Path to a WAV/MP3/M4A audio file")
    parser.add_argument(
        "--whisper-model",
        default=None,
        help="Whisper model size, for example tiny, base, small, medium, large",
    )
    args = parser.parse_args()

    print(json.dumps(analyze_voice(args.audio_path, args.whisper_model), indent=2))
