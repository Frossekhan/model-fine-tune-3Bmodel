# Qwen 3B Sentiment Analysis Fine-tuning

This module provides fine-tuning capabilities for Qwen 2.5 3B Instruct model on sentiment analysis using the `GerindT/mini_amazon_sentimental` dataset.

## Overview

The existing sentiment analysis uses a Multinomial Naive Bayes classifier. This new module adds the ability to fine-tune a Qwen 3B language model for sentiment analysis using LoRA (Low-Rank Adaptation).

## Files Added/Modified

### New Files
- `training/qwen_sentiment_trainer.py` - Main trainer class for Qwen 3B fine-tuning
- `training/train_qwen_sentiment.py` - CLI script to run training
- `scripts/run_qwen_sentiment_training.sh` - Shell script for easy training execution
- `README_QWEN_SENTIMENT.md` - This documentation

### Modified Files
- `requirements.txt` - Added `datasets>=2.14.0` and `evaluate>=0.4.0`
- `app/config.py` - Added `sentiment_model_base` and `sentiment_lora_output_dir` settings

## Prerequisites

1. **GPU Required**: Training requires a GPU with at least 8GB VRAM (16GB+ recommended)
2. **Dependencies**: Install requirements
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option 1: Using the Shell Script (Recommended)

```bash
# Basic usage with default settings
bash scripts/run_qwen_sentiment_training.sh

# Custom output directory and steps
bash scripts/run_qwen_sentiment_training.sh ./my_model 200
```

### Option 2: Using Python Directly

```bash
# Basic usage
python training/train_qwen_sentiment.py

# Custom parameters
python training/train_qwen_sentiment.py \
    --output-dir ./models/sentiment_qwen_3b_lora \
    --max-steps 100
```

### Option 3: Using as a Module

```python
from training.qwen_sentiment_trainer import QwenSentimentTrainer

# Initialize trainer
trainer = QwenSentimentTrainer(output_dir="./models/sentiment_qwen_3b_lora")

# Train the model
trainer.train(max_steps=100)
```

## Configuration

Environment variables (or settings in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `SENTIMENT_MODEL_BASE` | `Qwen/Qwen2.5-3B-Instruct` | Base model to fine-tune |
| `SENTIMENT_LORA_OUTPUT_DIR` | `./models/sentiment_qwen_3b_lora` | Where to save LoRA adapter |
| `FINE_TUNE_MAX_STEPS` | `10` | Maximum training steps |
| `FINE_TUNE_LEARNING_RATE` | `2e-5` | Learning rate |
| `MAX_TOKENS` | `1024` | Maximum sequence length |

## Dataset

The trainer automatically loads the dataset from Hugging Face:
- **Source**: `GerindT/mini_amazon_sentimental`
- **Format**: Automatically converted to instruction format
- **Task**: Sentiment analysis (positive/negative/neutral)

### Dataset Format

The dataset is automatically formatted as:
```
System: You are a sentiment analysis assistant. Analyze the sentiment of the given text and respond with only one word: positive, negative, or neutral.
User: Analyze the sentiment of this text: [review text]
Assistant: [sentiment label]
```

## Model Architecture

- **Base Model**: Qwen/Qwen2.5-3B-Instruct
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Target Modules**: `q_proj`, `v_proj`, `k_proj`, `o_proj`
- **LoRA Rank**: 8
- **LoRA Alpha**: 16
- **Dropout**: 0.05

## Training Parameters

Default training configuration:
- **Batch Size**: 4 per device
- **Gradient Accumulation**: 4 steps
- **Learning Rate**: 2e-5
- **Max Steps**: 100 (configurable)
- **FP16**: Enabled (if CUDA available)
- **Optimizer**: AdamW (via Trainer)

## Output

After training, the following files are saved to the output directory:
- `adapter_config.json` - LoRA configuration
- `adapter_model.safetensors` - LoRA weights
- `tokenizer_config.json` - Tokenizer configuration
- `special_tokens_map.json` - Special tokens
- `tokenizer.json` - Full tokenizer

## Using the Fine-tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./models/sentiment_qwen_3b_lora")
model.eval()

# Inference
messages = [
    {"role": "system", "content": "You are a sentiment analysis assistant. Analyze the sentiment of the given text and respond with only one word: positive, negative, or neutral."},
    {"role": "user", "content": "Analyze the sentiment of this text: This product is amazing! I love it!"}
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10)
    
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Integration with Existing System

The fine-tuned Qwen 3B model can be integrated into the existing sentiment service by:

1. Creating a new service class in `app/services/qwen_sentiment_service.py`
2. Updating `app/main.py` to initialize the new service
3. Adding API endpoints in `app/api/v1/routes.py`

## Troubleshooting

### Out of Memory (OOM)
- Reduce `per_device_train_batch_size` in `qwen_sentiment_trainer.py`
- Reduce `max_length` in config
- Use gradient checkpointing (add to LoraConfig)

### Slow Training
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU utilization: `nvidia-smi`
- Consider using a smaller base model or fewer training steps

### Dataset Loading Issues
- Ensure internet connection for first-time dataset download
- Check Hugging Face cache directory permissions
- Verify dataset name: `GerindT/mini_amazon_sentimental`

## Performance Expectations

- **Training Time**: ~30-60 minutes for 100 steps on RTX 3090/4090
- **Model Size**: ~3B parameters (base) + ~20MB (LoRA adapter)
- **Inference Speed**: ~50-100 tokens/second on GPU

## Next Steps

1. Run initial training with default settings
2. Evaluate model performance on validation set
3. Adjust hyperparameters (learning rate, LoRA rank, etc.)
4. Test inference with real-world examples
5. Integrate with existing API endpoints