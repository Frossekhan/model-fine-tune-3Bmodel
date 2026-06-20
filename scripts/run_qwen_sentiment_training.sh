#!/bin/bash
set -e

# Default values
OUTPUT_DIR=${1:-./models/sentiment_qwen_3b_lora}
MAX_STEPS=${2:-100}

echo "=========================================="
echo "Qwen 3B Sentiment Fine-tuning"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Max steps: $MAX_STEPS"
echo "Dataset: GerindT/mini_amazon_sentimental"
echo "=========================================="

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: No GPU detected. Training will run on CPU (very slow)."
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
python training/train_qwen_sentiment.py \
    --output-dir "$OUTPUT_DIR" \
    --max-steps "$MAX_STEPS"

echo "=========================================="
echo "Training completed!"
echo "LoRA adapter saved to: $OUTPUT_DIR"
echo "=========================================="