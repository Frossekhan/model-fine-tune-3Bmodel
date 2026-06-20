import logging
import hashlib
import math
import torch
from transformers import AutoModel, AutoTokenizer
from app.config import settings

logger = logging.getLogger(__name__)

FALLBACK_EMBEDDING_DIM = 384


class EmbeddingsService:
    def __init__(self):
        self.tokenizer = None
        self.model = None

    def initialize(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
            self.model = AutoModel.from_pretrained(settings.embedding_model)
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
            logger.info("Embedding model loaded: %s", settings.embedding_model)
        except Exception as e:
            logger.warning("Failed to initialize embedding model: %s. RAG will not work.", str(e))
            self.tokenizer = None
            self.model = None

    @torch.inference_mode()
    def embed(self, texts):
        if self.tokenizer is None or self.model is None:
            if isinstance(texts, str):
                texts = [texts]
            logger.warning("Embedding model not initialized. Using hashing embeddings.")
            return [self._hash_embed(text) for text in texts]
        
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        outputs = self.model(**inputs)
        vector = outputs.last_hidden_state[:, 0, :]
        return vector.detach().cpu().numpy().tolist()

    def _hash_embed(self, text: str):
        vector = [0.0] * FALLBACK_EMBEDDING_DIM
        tokens = str(text).lower().split()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % FALLBACK_EMBEDDING_DIM
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


embeddings_service = EmbeddingsService()
