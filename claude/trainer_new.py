#!/usr/bin/env python3
"""
Training pipeline for SaulLM Question Answering model
Uses quantized model with LoRA adapters for efficient fine-tuning
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling,
    TrainerCallback,
    EarlyStoppingCallback
)
from datasets import Dataset as HFDataset
import json
import os
import logging
from typing import Dict, List, Optional, Union
from model import SaulLMQuestionAnswering

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalQADataset(Dataset):
    """
    Custom dataset for legal question answering fine-tuning.
    Formats data for causal language modeling with instruction following.
    """
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        logger.info(f"Loading dataset from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} legal consultation examples")
        
        # Preprocess data into instruction format
        self.processed_data = self._preprocess_data()
    
    def _preprocess_data(self) -> List[str]:
        """Preprocess data into instruction-following format"""
        processed = []
        
        for item in self.data:
            # Create instruction-following format
            if "context" in item and item.get("context", "").strip():
                instruction = f"""[LEGAL_Q] Pergunta sobre imigração em Portugal: {item['question']}
[CONTEXT] Contexto: {item['context']}
[LEGAL_A] Resposta detalhada: {item['answer']}{self.tokenizer.eos_token}"""
            else:
                instruction = f"""[LEGAL_Q] Pergunta sobre imigração em Portugal: {item['question']}
[LEGAL_A] Resposta detalhada: {item['answer']}{self.tokenizer.eos_token}"""
            
            processed.append(instruction)
        
        return processed
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        text = self.processed_data[idx]
        
        # Tokenize the complete instruction
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # For causal LM, input_ids and labels are the same
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()  # For causal LM training
        }

class SaulLMTrainer:
    """
    Trainer class for SaulLM legal question answering model.
    Handles training with quantization and LoRA adapters.
    """
    
    def __init__(self, 
                 model_name: str = "Equall/Saul-7B-Instruct-v1",
                 use_quantization: bool = True,
                 use_peft: bool = True,
                 output_dir: str = "./legal_saul_model"):
        """
        Initialize the trainer.
        
        Args:
            model_name: Base model name
            use_quantization: Whether to use 4-bit quantization
            use_peft: Whether to use LoRA adapters
            output_dir: Output directory for saving model
        """
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.use_peft = use_peft
        self.output_dir = output_dir
        
        # Initialize model
        logger.info("Initializing SaulLM model...")
        self.model_wrapper = SaulLMQuestionAnswering(
            model_name=model_name,
            use_quantization=use_quantization,
            use_peft=use_peft
        )
        
        self.model = self.model_wrapper.get_model()
        self.tokenizer = self.model_wrapper.get_tokenizer()
        
        # Print model info
        info = self.model_wrapper.get_model_info()
        logger.info("Model Information:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")
    
    def prepare_dataset(self, data_path: str, max_length: int = 1024) -> LegalQADataset:
        """Prepare dataset for training"""
        logger.info(f"Preparing dataset from {data_path}")
        dataset = LegalQADataset(data_path, self.tokenizer, max_length)
        return dataset
    
    def train(self, 
              data_path: str,
              output_dir: Optional[str] = None,
              num_train_epochs: int = 3,
              per_device_train_batch_size: int = 1,
              gradient_accumulation_steps: int = 4,
              learning_rate: float = 2e-4,
              warmup_steps: int = 100,
              logging_steps: int = 10,
              save_steps: int = 500,
              eval_steps: int = 500,
              max_length: int = 1024,
              eval_data_path: Optional[str] = None,
              **kwargs):
        """
        Train the model.
        
        Args:
            data_path: Path to training data
            output_dir: Output directory (overrides default)
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            learning_rate: Learning rate
            warmup_steps: Warmup steps
            logging_steps: Logging frequency
            save_steps: Save frequency
            eval_steps: Evaluation frequency
            max_length: Maximum sequence length
            eval_data_path: Path to evaluation data
            **kwargs: Additional training arguments
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        # Prepare dataset
        train_dataset = self.prepare_dataset(data_path, max_length)
        eval_dataset = None
        if eval_data_path:
            eval_dataset = self.prepare_dataset(eval_data_path, max_length)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=eval_steps if eval_dataset else None,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            report_to=["tensorboard"],
            logging_dir=f"{output_dir}/logs",
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            **kwargs
        )
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if eval_dataset else None,
        )
        
        # Train the model
        logger.info("Starting training...")
        trainer.train()
        
        # Save the final model
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info("Training completed!")
        return trainer
    
    def evaluate(self, data_path: str, max_length: int = 1024):
        """Evaluate the model on a dataset"""
        logger.info(f"Evaluating model on {data_path}")
        
        eval_dataset = self.prepare_dataset(data_path, max_length)
        
        # Simple evaluation using model perplexity
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(len(eval_dataset)):
                batch = eval_dataset[i]
                inputs = {k: v.unsqueeze(0) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].unsqueeze(0)
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                    labels = labels.cuda()
                
                outputs = self.model(**inputs, labels=labels)
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        logger.info(f"Evaluation Results:")
        logger.info(f"  Average Loss: {avg_loss:.4f}")
        logger.info(f"  Perplexity: {perplexity:.4f}")
        
        return {"loss": avg_loss, "perplexity": perplexity.item()}
    
    def generate_answer(self, question: str, context: str = "", **generation_kwargs):
        """Generate answer for a question"""
        return self.model_wrapper.generate_answer(question, context, **generation_kwargs)
    
    @classmethod
    def load_trained_model(cls, model_path: str, base_model_name: str = "Equall/Saul-7B-Instruct-v1"):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
            base_model_name: Base model name
            
        Returns:
            Trainer instance with loaded model
        """
        logger.info(f"Loading trained model from {model_path}")
        
        # Create trainer instance
        trainer = cls(model_name=base_model_name, use_peft=True)
        
        # Load the trained model
        trainer.model_wrapper = SaulLMQuestionAnswering.load_trained_model(
            model_path, base_model_name
        )
        trainer.model = trainer.model_wrapper.get_model()
        trainer.tokenizer = trainer.model_wrapper.get_tokenizer()
        
        return trainer

# Convenience function for quick training
def train_legal_qa_model(
    data_path: str,
    output_dir: str = "./legal_saul_model",
    model_name: str = "Equall/Saul-7B-Instruct-v1",
    num_epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    use_quantization: bool = True,
    **kwargs
):
    """
    Convenience function to train a legal QA model.
    
    Args:
        data_path: Path to training data JSON file
        output_dir: Output directory for saving model
        model_name: Base model name
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        use_quantization: Whether to use 4-bit quantization
        **kwargs: Additional training arguments
        
    Returns:
        Trained model trainer
    """
    # Initialize trainer
    trainer = SaulLMTrainer(
        model_name=model_name,
        use_quantization=use_quantization,
        use_peft=True,
        output_dir=output_dir
    )
    
    # Train the model
    trainer.train(
        data_path=data_path,
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        **kwargs
    )
    
    return trainer
