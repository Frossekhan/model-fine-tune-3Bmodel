import os
from pathlib import Path
from typing import Dict

from app.services.emotion_model import EmotionModel
from app.services.nlp_utils import preprocess_text


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


def analyze_voice(audio_path: str, model_size: str | None = None) -> Dict[str, object]:
    transcript = transcribe_audio(audio_path, model_size=model_size)
    processed_text = preprocess_text(transcript)
    sentiment = EmotionModel.analyze(processed_text)

    return {
        "audio_path": str(audio_path),
        "transcript": transcript,
        "processed_text": processed_text,
        "sentiment": sentiment["sentiment"],
        "confidence": sentiment["confidence"],
        "probabilities": sentiment["probabilities"],
        "model": sentiment["model"],
    }


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
