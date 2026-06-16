import re


_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_PATTERN = re.compile(r"@\w+")
_HASHTAG_PATTERN = re.compile(r"#(\w+)")
_SPACE_PATTERN = re.compile(r"\s+")
_PUNCT_PATTERN = re.compile(r"[^a-z0-9\s']")


def clean_text(text: str) -> str:
    """Normalize casual speech/text before sentiment analysis."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = text.lower().strip()
    normalized = _URL_PATTERN.sub(" ", normalized)
    normalized = _MENTION_PATTERN.sub(" ", normalized)
    normalized = _HASHTAG_PATTERN.sub(r"\1", normalized)
    normalized = _PUNCT_PATTERN.sub(" ", normalized)
    normalized = _SPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


def preprocess_text(text: str) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("text is empty after preprocessing")
    return cleaned
