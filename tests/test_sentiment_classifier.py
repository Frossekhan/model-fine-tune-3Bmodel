import json

import pytest

from app.api.v1.schemas import SentimentResponse
from app.services.sentiment_service import SentimentService
from training.sentiment_classifier import MultinomialNaiveBayes, tokenize


TRAIN_TEXTS = [
    "I love this excellent service",
    "This product is wonderful",
    "I hate this terrible service",
    "This product is awful",
    "The package arrived today",
    "The meeting starts at noon",
]
TRAIN_LABELS = [
    "positive",
    "positive",
    "negative",
    "negative",
    "neutral",
    "neutral",
]


def test_tokenizer_marks_words_after_negation():
    assert tokenize("not very good today") == [
        "not",
        "not_very",
        "not_good",
        "not_today",
    ]


def test_classifier_predicts_each_sentiment():
    model = MultinomialNaiveBayes().fit(TRAIN_TEXTS, TRAIN_LABELS)

    assert model.predict("excellent wonderful service")[0] == "positive"
    assert model.predict("awful terrible product")[0] == "negative"
    assert model.predict("the meeting arrived at noon")[0] == "neutral"


def test_probabilities_sum_to_one():
    model = MultinomialNaiveBayes().fit(TRAIN_TEXTS, TRAIN_LABELS)
    probabilities = model.predict_proba("I love the product")

    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_unseen_vocabulary_falls_back_to_neutral():
    model = MultinomialNaiveBayes().fit(TRAIN_TEXTS, TRAIN_LABELS)

    assert model.predict("xyzzy plugh")[0] == "neutral"


def test_model_round_trip(tmp_path):
    model_path = tmp_path / "sentiment.json"
    original = MultinomialNaiveBayes().fit(TRAIN_TEXTS, TRAIN_LABELS)
    original.save(str(model_path))

    loaded = MultinomialNaiveBayes.load(str(model_path))

    assert loaded.predict("excellent service")[0] == "positive"
    assert json.loads(model_path.read_text())["model_type"] == (
        "multinomial_naive_bayes"
    )


def test_sentiment_service_response_shape():
    SentimentService.classifier = MultinomialNaiveBayes().fit(
        TRAIN_TEXTS,
        TRAIN_LABELS,
    )

    result = SentimentService.analyze("terrible and awful")

    assert result["sentiment"] == "negative"
    assert result["model"] == "multinomial_naive_bayes"
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["probabilities"]) == {"negative", "neutral", "positive"}


def test_sentiment_response_schema():
    SentimentService.classifier = MultinomialNaiveBayes().fit(
        TRAIN_TEXTS,
        TRAIN_LABELS,
    )

    response = SentimentResponse(
        **SentimentService.analyze("wonderful excellent product")
    )

    assert isinstance(response, SentimentResponse)
    assert response.sentiment == "positive"
