import os
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
    app_name: str = "Enterprise AI Assistant"
    environment: str = Field(default="production", env="ENVIRONMENT")
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    model_base: str = Field(default="Qwen/Qwen2.5-7B-Instruct", env="MODEL_BASE")
    lora_output_dir: str = Field(default="", env="LORA_OUTPUT_DIR")
    adapter_name: str = Field(default="qwen-7b-lora-final", env="LORA_ADAPTER_NAME")
    sentiment_model_base: str = Field(default="Qwen/Qwen2.5-3B-Instruct", env="SENTIMENT_MODEL_BASE")
    sentiment_lora_output_dir: str = Field(default="./models/sentiment_qwen_3b_lora", env="SENTIMENT_LORA_OUTPUT_DIR")
    embedding_model: str = Field(default="BAAI/bge-large-en-v1.5", env="EMBEDDING_MODEL")
    chroma_persist_dir: str = Field(default="./chroma_store_bge", env="CHROMA_PERSIST_DIR")
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    api_key: str = Field(default="", env="API_KEY")
    max_history: int = Field(default=20, env="MAX_HISTORY")
    max_tokens: int = Field(default=1024, env="MAX_TOKENS")
    max_new_tokens: int = Field(default=512, env="MAX_NEW_TOKENS")
    fine_tune_max_steps: int = Field(default=10, env="FINE_TUNE_MAX_STEPS")
    fine_tune_learning_rate: float = Field(default=2e-5, env="FINE_TUNE_LEARNING_RATE")
    sentiment_model_path: str = Field(
        default="./models/sentiment/naive_bayes.json",
        env="SENTIMENT_MODEL_PATH",
    )
    sentiment_dataset_path: str = Field(
        default="./training/examples/sentiment_dataset.json",
        env="SENTIMENT_DATASET_PATH",
    )
    sentiment_alpha: float = Field(default=1.0, env="SENTIMENT_ALPHA")
    temperature: float = Field(default=0.3, env="TEMPERATURE")
    top_p: float = Field(default=0.95, env="TOP_P")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    service_timeout: int = Field(default=60, env="SERVICE_TIMEOUT")
    enable_streaming: bool = Field(default=True, env="ENABLE_STREAMING")


settings = Settings()
