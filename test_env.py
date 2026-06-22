import sys
print("Python version:", sys.version)

try:
    import torch
    print("torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
except Exception as e:
    print("torch import error:", e)

try:
    import transformers
    print("transformers version:", transformers.__version__)
except Exception as e:
    print("transformers import error:", e)

try:
    import peft
    print("peft version:", peft.__version__)
except Exception as e:
    print("peft import error:", e)

try:
    import datasets
    print("datasets version:", datasets.__version__)
except Exception as e:
    print("datasets import error:", e)

try:
    from app.config import settings
    print("Settings loaded successfully")
    print("Sentiment model base:", settings.sentiment_model_base)
    print("LoRA output dir:", settings.sentiment_lora_output_dir)
except Exception as e:
    print("config import error:", e)