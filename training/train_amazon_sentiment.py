import json
from pathlib import Path
from datasets import load_dataset
import sys

# Resolve paths relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Load the dataset
print("Loading dataset GerindT/mini_amazon_sentimental...")
ds = load_dataset("GerindT/mini_amazon_sentimental")

print("Dataset loaded:")
print(ds)

# Inspect the structure
print("\nFirst example:")
print(ds['test'][0])

# Convert to instruction format for LoRA training
# Assuming the dataset has 'text' and 'label' columns
# We'll create instruction/input/output format

import ast

def convert_to_instruction_format(example):
    """Convert dataset example to instruction format."""
    text = example.get('text', '')
    label = example.get('label', '')
    sentiment = example.get('sentiment', '')
    
    # Parse the label field which is a string representation of a dict
    try:
        label_dict = ast.literal_eval(label) if isinstance(label, str) else label
        if isinstance(label_dict, dict):
            label = label_dict.get('label', sentiment)
    except Exception:
        label = sentiment if sentiment else label
    
    # Map star ratings to sentiment
    label_str = str(label).lower()
    if '5' in label_str or '4' in label_str or 'perfect' in label_str or 'excellent' in label_str:
        sentiment_label = 'positive'
    elif '1' in label_str or '2' in label_str or 'terrible' in label_str or 'awful' in label_str:
        sentiment_label = 'negative'
    else:
        sentiment_label = 'neutral'
    
    return {
        "instruction": "Classify the sentiment of the following customer review.",
        "input": text,
        "output": f"Sentiment: {sentiment_label}"
    }

# Process the dataset
train_data = ds['test']
limit = 10000
if len(sys.argv) > 1:
    limit = int(sys.argv[1])
print(f"Using {limit} examples out of {len(train_data)}")
train_data = train_data.select(range(min(limit, len(train_data))))
instruction_data = [convert_to_instruction_format(ex) for ex in train_data]

# Save to JSON
output_path = PROJECT_ROOT / "training/examples/amazon_sentiment_train.json"
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(instruction_data, f, indent=2, ensure_ascii=False)

print(f"\nConverted {len(instruction_data)} examples")
print(f"Saved to: {output_path}")
print("\nFirst 3 converted examples:")
for i, ex in enumerate(instruction_data[:3]):
    print(f"\n{i+1}. Instruction: {ex['instruction']}")
    print(f"   Input: {ex['input'][:100]}...")
    print(f"   Output: {ex['output']}")