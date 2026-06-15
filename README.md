# Enterprise AI Assistant Backend

A FastAPI-based backend for an enterprise-grade AI assistant built around a fine-tuned Qwen2.5-1.5B-Instruct model with LoRA, RAG, ChromaDB retrieval, Redis memory, tool calling, realtime streaming, and Naive Bayes text sentiment analysis.

## Architecture

- `app/` - production API and service code
- `training/` - fine-tuning dataset and trainer pipelines
- `deployment/` - Docker deployment manifests
- `scripts/` - run scripts for training and serving

## Setup

1. Create a Python environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Configure environment variables in `.env`:
   - `MODEL_BASE` (e.g. `Qwen2.5-1.5B-Instruct` or your Hugging Face path)
   - `LORA_OUTPUT_DIR`
   - `CHROMA_PERSIST_DIR`
   - `REDIS_URL`
   - `EMBEDDING_MODEL`
   - `API_HOST`, `API_PORT`

3. Start Redis and ChromaDB services.

4. Train or load adapters and run the API.

## Sentiment Analysis

The sentiment feature uses a persisted Multinomial Naive Bayes classifier and
returns `positive`, `negative`, or `neutral`.

Train it with the included dataset:

```bash
python training/train_sentiment.py
```

The server also trains and saves the model automatically when the configured
model file does not exist.

Analyze text:

```bash
curl -X POST http://localhost:8000/api/v1/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text":"The support team was excellent"}'
```

Example response:

```json
{
  "text": "The support team was excellent",
  "sentiment": "positive",
  "confidence": 0.91,
  "probabilities": {
    "negative": 0.04,
    "neutral": 0.05,
    "positive": 0.91
  },
  "model": "multinomial_naive_bayes"
}
```

## Deployment

- Build container: `docker build -t enterprise-ai-assistant .`
- Run compose: `docker compose up --build`

## Notes

This repository includes:
- LoRA fine-tuning pipeline
- Instruction-style dataset creation
- Tokenization pipeline
- RAG retrieval service
- Redis memory store
- Tool-calling architecture
- Streaming REST API with monitoring
- Multinomial Naive Bayes sentiment training and prediction API
