"""
Quick verification script for Qwen sentiment training module.
Tests imports and basic functionality without requiring GPU.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from app.config import settings
        print("✓ app.config imported successfully")
    except Exception as e:
        print(f"✗ Failed to import app.config: {e}")
        return False
    
    try:
        from training.qwen_sentiment_trainer import QwenSentimentTrainer
        print("✓ training.qwen_sentiment_trainer imported successfully")
    except Exception as e:
        print(f"✗ Failed to import QwenSentimentTrainer: {e}")
        return False
    
    try:
        from training.train_qwen_sentiment import train_qwen_sentiment_model
        print("✓ training.train_qwen_sentiment imported successfully")
    except Exception as e:
        print(f"✗ Failed to import train_qwen_sentiment_model: {e}")
        return False
    
    return True


def test_config():
    """Test configuration settings."""
    print("\nTesting configuration...")
    
    from app.config import settings
    
    print(f"  Base model: {settings.sentiment_model_base}")
    print(f"  LoRA output dir: {settings.sentiment_lora_output_dir}")
    print(f"  Max tokens: {settings.max_tokens}")
    print(f"  Learning rate: {settings.fine_tune_learning_rate}")
    print(f"  Max steps: {settings.fine_tune_max_steps}")
    
    if settings.sentiment_model_base != "Qwen/Qwen2.5-3B-Instruct":
        print("✗ sentiment_model_base not set correctly")
        return False
    
    print("✓ Configuration looks good")
    return True


def test_dataset_loading():
    """Test dataset loading (requires internet)."""
    print("\nTesting dataset loading...")
    
    try:
        from datasets import load_dataset
        print("  Loading GerindT/mini_amazon_sentimental (this may take a moment)...")
        
        # Just test that we can access the dataset info
        dataset = load_dataset("GerindT/mini_amazon_sentimental", split="train", streaming=True)
        
        # Get first example
        first_example = next(iter(dataset))
        print(f"  Dataset loaded successfully!")
        print(f"  First example keys: {list(first_example.keys())}")
        print(f"  First example: {first_example}")
        
        print("✓ Dataset loading works")
        return True
        
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        print("  Note: This is expected if you don't have internet or the dataset is not accessible")
        return False


def test_trainer_initialization():
    """Test that trainer can be initialized (without loading model)."""
    print("\nTesting trainer initialization...")
    
    try:
        from training.qwen_sentiment_trainer import QwenSentimentTrainer
        
        # This will download the tokenizer
        print("  Initializing QwenSentimentTrainer (downloading tokenizer)...")
        trainer = QwenSentimentTrainer(output_dir="./test_output")
        
        print(f"  Model name: {trainer.model_name}")
        print(f"  Tokenizer: {trainer.tokenizer.__class__.__name__}")
        print(f"  Pad token: {trainer.tokenizer.pad_token}")
        
        print("✓ Trainer initialized successfully")
        return True
        
    except Exception as e:
        print(f"✗ Failed to initialize trainer: {e}")
        return False


def main():
    print("=" * 60)
    print("Qwen Sentiment Training Module - Verification Tests")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Dataset Loading", test_dataset_loading()))
    results.append(("Trainer Initialization", test_trainer_initialization()))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 60)
    if all_passed:
        print("All tests passed! ✓")
        print("\nYou can now run training with:")
        print("  bash scripts/run_qwen_sentiment_training.sh")
        print("  or")
        print("  python training/train_qwen_sentiment.py")
    else:
        print("Some tests failed. Please check the errors above.")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())