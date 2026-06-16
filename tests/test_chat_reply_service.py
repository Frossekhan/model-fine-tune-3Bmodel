from app.services.chat_reply_service import ChatReplyService


def test_model_not_loaded_response_uses_local_fallback():
    assert ChatReplyService.needs_fallback("[Model not loaded] Echo: hello")


def test_fallback_reply_includes_sentiment_context():
    sentiment = {
        "sentiment": "negative",
        "confidence": 0.91,
        "probabilities": {"negative": 0.91, "neutral": 0.05, "positive": 0.04},
        "model": "multinomial_naive_bayes",
        "text": "I am stressed",
    }

    reply = ChatReplyService.build_fallback("I am stressed, what should I do?", sentiment)

    assert "You asked:" in reply
    assert "negative" in reply
    assert "supportive" in reply
