import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging
from app.config import settings
from training.qwen_sentiment_trainer import QwenSentimentTrainer

logger = logging.getLogger(__name__)


def train_qwen_sentiment_model(output_dir: str = None, max_steps: int = None):
    """Train Qwen 3B model on sentiment analysis using mini_amazon_sentimental dataset."""
    output_dir = output_dir or settings.lora_output_dir
    max_steps = max_steps or getattr(settings, "fine_tune_max_steps", 100)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("Initializing Qwen 3B sentiment trainer")
    trainer = QwenSentimentTrainer(output_dir=output_dir)
    
    logger.info("Starting training with dataset: GerindT/mini_amazon_sentimental")
    trainer.train(max_steps=max_steps)
    
    logger.info("Training completed. Model saved to: %s", output_dir)
    print(f"Training completed. LoRA adapter saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Qwen 3B model for sentiment analysis using mini_amazon_sentimental dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for LoRA adapter (default: from settings)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps (default: from settings or 100)",
    )
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    train_qwen_sentiment_model(
        output_dir=args.output_dir,
        max_steps=args.max_steps
    )