import logging
from pathlib import Path
from typing import Dict

from app.config import settings
from training.sentiment_classifier import (
    MultinomialNaiveBayes,
    load_sentiment_dataset,
)


logger = logging.getLogger(__name__)


class SentimentService:
    classifier: MultinomialNaiveBayes = None

    @classmethod
    def initialize(cls) -> None:
        model_path = Path(__file__).resolve().parents[2] / settings.sentiment_model_path
        if model_path.exists():
            cls.classifier = MultinomialNaiveBayes.load(str(model_path))
            logger.info("Loaded sentiment model from %s", model_path)
            return

        dataset_path = Path(__file__).resolve().parents[2] / settings.sentiment_dataset_path
        texts, labels = load_sentiment_dataset(str(dataset_path))
        cls.classifier = MultinomialNaiveBayes(alpha=settings.sentiment_alpha)
        cls.classifier.fit(texts, labels)
        cls.classifier.save(str(model_path))
        logger.info(
            "Trained sentiment model with %d records and saved it to %s",
            len(texts),
            model_path,
        )

    @classmethod
    def analyze(cls, text: str) -> Dict[str, object]:
        if cls.classifier is None:
            cls.initialize()
        label, confidence, probabilities = cls.classifier.predict(text)
        return {
            "text": text,
            "sentiment": label,
            "confidence": round(confidence, 6),
            "probabilities": {
                name: round(probability, 6)
                for name, probability in probabilities.items()
            },
            "model": "multinomial_naive_bayes",
        }
