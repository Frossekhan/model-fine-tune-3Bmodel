import os
import json
import logging
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from app.config import settings

logger = logging.getLogger(__name__)


class QwenSentimentTrainer:
    def __init__(self, output_dir: str = None, model_name: str = None):
        self.output_dir = output_dir or settings.lora_output_dir
        self.model_name = model_name or getattr(settings, "sentiment_model_base", "Qwen/Qwen2.5-3B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_and_prepare_dataset(self):
        """Load the mini_amazon_sentimental dataset and prepare for training."""
        logger.info("Loading dataset GerindT/mini_amazon_sentimental")
        dataset = load_dataset("GerindT/mini_amazon_sentimental", split="test")
        
        # Convert to instruction format for sentiment analysis
        def format_example(example):
            text = example.get("text", example.get("review", ""))
            label = example.get("label", example.get("sentiment", ""))
            
            # Map labels to sentiment categories if needed
            if isinstance(label, int):
                label_map = {0: "negative", 1: "positive", 2: "neutral"}
                label = label_map.get(label, str(label))
            else:
                label = str(label).lower()
            
            system_prompt = "You are a sentiment analysis assistant. Analyze the sentiment of the given text and respond with only one word: positive, negative, or neutral."
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze the sentiment of this text: {text}"},
                {"role": "assistant", "content": label}
            ]
            
            return {
                "text": text,
                "label": label,
                "messages": messages
            }
        
        formatted_dataset = dataset.map(format_example, remove_columns=dataset.column_names)
        
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                self.tokenizer.apply_chat_template(
                    examples["messages"],
                    tokenize=False
                ),
                truncation=True,
                max_length=settings.max_tokens,
                padding="max_length",
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        logger.info("Tokenizing dataset")
        tokenized_dataset = formatted_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=formatted_dataset.column_names
        )
        
        return tokenized_dataset

    def configure_peft(self):
        """Configure LoRA parameters for Qwen 3B."""
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        return peft_config

    def train(self, max_steps: int = None):
        """Train the Qwen 3B model on sentiment analysis."""
        max_steps = max_steps or getattr(settings, "fine_tune_max_steps", 100)
        
        logger.info("Loading base model: %s", self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        
        # Apply LoRA
        logger.info("Configuring LoRA adapter")
        peft_model = get_peft_model(model, self.configure_peft())
        peft_model.print_trainable_parameters()
        
        # Load and prepare dataset
        train_dataset = self.load_and_prepare_dataset()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            max_steps=max_steps,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=settings.fine_tune_learning_rate,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            save_steps=max_steps // 4,
            save_total_limit=2,
            report_to="none",
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=peft_model,
            padding=True,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Train
        logger.info("Starting training for %d steps", max_steps)
        trainer.train()
        
        # Save model
        logger.info("Saving LoRA adapter to %s", self.output_dir)
        peft_model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info("Training completed successfully")
        return trainer