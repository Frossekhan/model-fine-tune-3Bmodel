from __future__ import annotations

import re
from typing import Any, Dict


class ChatReplyService:
    model_not_loaded_prefix = "[Model not loaded]"

    @classmethod
    def needs_fallback(cls, response: str | None) -> bool:
        if not response or not response.strip():
            return True
        return response.strip().startswith(cls.model_not_loaded_prefix)

    @staticmethod
    def build_fallback(prompt: str, sentiment: Dict[str, Any]) -> str:
        clean_prompt = " ".join(prompt.split())
        lower_prompt = clean_prompt.lower()
        sentiment_label = str(sentiment.get("sentiment", "neutral")).lower()

        if ChatReplyService._is_greeting(lower_prompt):
            answer = (
                "Hi! I am ready. Ask me a question and I will reply, analyze "
                "the sentiment, and speak the answer aloud."
            )
        elif "how are you" in lower_prompt:
            answer = "I am ready and working. Tell me what you want to know."
        elif "your name" in lower_prompt or "who are you" in lower_prompt:
            answer = (
                "I am your Enterprise AI Assistant. I can answer your messages, "
                "classify their sentiment, and read my replies aloud."
            )
        elif "what can you do" in lower_prompt:
            answer = (
                "I can chat with you, answer common questions, detect whether "
                "your message is positive, negative, or neutral, and speak the "
                "reply in the browser."
            )
        elif ChatReplyService._looks_like_question(clean_prompt):
            answer = (
                f"You asked: {clean_prompt} I will answer based on your message. "
                "The local full language model is not loaded right now, so I am "
                "using the built-in chat reply mode."
            )
        else:
            answer = (
                f"I understood your message: {clean_prompt} I will respond based "
                "on what you typed."
            )

        return f"{answer} {ChatReplyService._sentiment_sentence(sentiment_label)}"

    @staticmethod
    def _is_greeting(text: str) -> bool:
        words = set(re.findall(r"[a-z]+", text))
        return bool(words.intersection({"hi", "hello", "hey", "vanakkam"}))

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        lower_text = text.lower().strip()
        question_starters = (
            "what",
            "why",
            "how",
            "who",
            "when",
            "where",
            "which",
            "can",
            "could",
            "should",
            "is",
            "are",
            "do",
            "does",
        )
        return lower_text.endswith("?") or lower_text.startswith(question_starters)

    @staticmethod
    def _sentiment_sentence(label: str) -> str:
        if label == "positive":
            return "Your message sounds positive."
        if label == "negative":
            return "Your message sounds negative, so I will keep the reply supportive."
        return "Your message sounds neutral."
