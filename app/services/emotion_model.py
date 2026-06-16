import logging
import os
from typing import Dict, List

from app.services.nlp_utils import preprocess_text


logger = logging.getLogger(__name__)


class EmotionModel:
    _pipeline = None
    _model_name = None

    @classmethod
    def initialize(cls) -> None:
        if cls._pipeline is not None:
            return

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for Hugging Face emotion analysis. "
                "Install project requirements first."
            ) from exc

        cls._model_name = os.getenv(
            "EMOTION_MODEL_NAME",
            "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        )
        cls._pipeline = pipeline(
            "sentiment-analysis",
            model=cls._model_name,
            tokenizer=cls._model_name,
            top_k=None,
        )
        logger.info("Loaded Hugging Face sentiment model: %s", cls._model_name)

    @classmethod
    def analyze(cls, text: str) -> Dict[str, object]:
        cleaned_text = preprocess_text(text)
        if cls._pipeline is None:
            cls.initialize()

        raw_result = cls._pipeline(cleaned_text)
        scores = _flatten_scores(raw_result)
        best = max(scores, key=lambda item: item["score"])
        probabilities = {
            _normalize_label(item["label"]): round(float(item["score"]), 6)
            for item in scores
        }

        return {
            "text": text,
            "processed_text": cleaned_text,
            "sentiment": _normalize_label(best["label"]),
            "confidence": round(float(best["score"]), 6),
            "probabilities": probabilities,
            "model": cls._model_name
            or os.getenv(
                "EMOTION_MODEL_NAME",
                "cardiffnlp/twitter-xlm-roberta-base-sentiment",
            ),
        }


def _flatten_scores(raw_result: object) -> List[Dict[str, object]]:
    if isinstance(raw_result, list) and raw_result:
        first = raw_result[0]
        if isinstance(first, list):
            return first
        if isinstance(first, dict):
            return raw_result
    raise RuntimeError(f"Unexpected sentiment model output: {raw_result!r}")


def _normalize_label(label: str) -> str:
    value = label.lower().strip()
    mapping = {
        "label_0": "negative",
        "label_1": "neutral",
        "label_2": "positive",
    }
    return mapping.get(value, value)
