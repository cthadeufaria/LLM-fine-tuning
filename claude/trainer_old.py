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
    TrainerCallback
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
    Custom dataset for legal consultation fine-tuning.
    Expects data in format: [{"question": "...", "context": "...", "answer": "..."}]
    """
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} legal consultation examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format the input for legal consultation
        if "context" in item and item["context"]:
            input_text = f"[LEGAL_Q] {item['question']} [IMMIGRATION] {item['context']}"
        else:
            input_text = f"[LEGAL_Q] {item['question']}"
        
        target_text = f"[LEGAL_A] {item['answer']}"
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": input_encoding["input_ids"].flatten(),
            "attention_mask": input_encoding["attention_mask"].flatten(),
            "labels": target_encoding["input_ids"].flatten()
        }

class LegalPEFTTrainer:
    """
    PEFT Fine-tuning trainer for Legal BERT Seq2Seq model
    """
    def __init__(
        self,
        model_name: str = "neuralmind/bert-base-portuguese-cased",
        use_peft: bool = True,
        output_dir: str = "./legal_model_results",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ):
        self.model_name = model_name
        self.use_peft = use_peft
        self.output_dir = output_dir
        
        # Initialize model and tokenizer
        self.model_wrapper = LegalBertSeq2Seq(
            bert_model_name=model_name,
            use_peft=use_peft,
            peft_config=self._create_peft_config(lora_r, lora_alpha, lora_dropout) if use_peft else None
        )
        
        self.model = self.model_wrapper.get_model()
        self.tokenizer = self.model_wrapper.get_tokenizer()
        
        # Print trainable parameters
        if use_peft:
            self.model_wrapper.print_trainable_parameters()
    
    def _create_peft_config(self, r: int, alpha: int, dropout: float):
        """Create PEFT configuration for BERT seq2seq"""
        from peft import LoraConfig, TaskType
        
        return LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["query", "value", "key", "dense"]  # BERT attention modules
        )
    
    def prepare_dataset(self, data_path: str, max_length: int = 512):
        """Prepare dataset for training"""
        dataset = LegalDataset(data_path, self.tokenizer, max_length)
        return dataset
    
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        num_train_epochs: int = 3,
        learning_rate: float = 5e-5,
        per_device_train_batch_size: int = 4,
        per_device_eval_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: int = 500,
        max_steps: int = -1,
        fp16: bool = True,
        gradient_checkpointing: bool = True,
    ):
        """Train the model with PEFT"""
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps if eval_dataset else None,
            eval_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            max_steps=max_steps,
            fp16=fp16,
            gradient_checkpointing=gradient_checkpointing,
            dataloader_drop_last=True,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            report_to=None,  # Disable wandb/tensorboard logging
            remove_unused_columns=False,
            predict_with_generate=True,  # For seq2seq generation
        )
        
        # Data collator for seq2seq
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
        )
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train the model
        print("Starting training...")
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"Training completed! Model saved to {self.output_dir}")
        
        return trainer
    
    def save_model(self, save_path: str):
        """Save the trained model"""
        if self.use_peft:
            self.model.save_pretrained(save_path)
        else:
            self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        from peft import PeftModel
        
        if self.use_peft:
            # Load base model first
            base_model = LegalBertSeq2Seq(
                bert_model_name=self.model_name,
                use_peft=False
            )
            # Load PEFT adapters
            self.model = PeftModel.from_pretrained(base_model.get_model(), model_path)
        else:
            self.model = LegalBertSeq2Seq(
                bert_model_name=model_path,
                use_peft=False
            ).get_model()
        
        print(f"Model loaded from {model_path}")

def create_legal_sample_data(output_path: str = "legal_sample_data.json"):
    """Create sample legal consultation data for Portuguese immigration law"""
    legal_data = [
        {
            "question": "Quanto tempo demora para obter um visto de trabalho em Portugal?",
            "context": "Cidadão brasileiro interessado em trabalhar em Portugal",
            "answer": "O prazo para obtenção de um visto de trabalho em Portugal varia entre 15 a 60 dias úteis, dependendo do tipo de visto e da documentação apresentada. É necessário ter uma proposta de trabalho de uma empresa portuguesa e apresentar documentos como contrato de trabalho, comprovativo de qualificações, certificado de antecedentes criminais e seguro de saúde."
        },
        {
            "question": "Quais são os requisitos para solicitar a nacionalidade portuguesa?",
            "context": "Residente em Portugal há 6 anos com título de residência",
            "answer": "Para solicitar a nacionalidade portuguesa, é necessário: 1) Residir legalmente em Portugal por pelo menos 5 anos; 2) Ter conhecimento suficiente da língua portuguesa; 3) Não ter sido condenado por crime punível com pena de prisão superior a 3 anos; 4) Ter vínculos efetivos à comunidade nacional. É necessário apresentar certificado de registo criminal, comprovativo de rendimentos e aprovação no exame de português."
        },
        {
            "question": "Como renovar o título de residência?",
            "context": "Título de residência expira em 2 meses",
            "answer": "A renovação do título de residência deve ser solicitada entre 30 dias antes e 90 dias após o prazo de validade. Deve dirigir-se ao SEF (atual AIMA) com os seguintes documentos: 1) Requerimento de renovação; 2) Documento de identificação válido; 3) Comprovativo de meios de subsistência; 4) Seguro de saúde; 5) Certificado de registo criminal português. A taxa de renovação é de €83,40."
        },
        {
            "question": "Posso trazer minha família para Portugal?",
            "context": "Tenho visto de trabalho válido em Portugal",
            "answer": "Sim, é possível solicitar reagrupamento familiar. Pode trazer cônjuge, filhos menores de 18 anos ou filhos maiores solteiros e dependentes, e ascendentes dependentes. É necessário comprovar: 1) Alojamento adequado; 2) Meios de subsistência suficientes; 3) Seguro de saúde; 4) Registo criminal limpo dos familiares. O pedido deve ser feito no consulado português no país de origem dos familiares."
        },
        {
            "question": "Qual a diferença entre autorização de residência e visto de residência?",
            "context": "Confuso sobre os tipos de documentos de residência",
            "answer": "A autorização de residência é concedida a quem já se encontra em Portugal com visto de entrada válido ou em situação legal, enquanto o visto de residência é solicitado no país de origem antes da viagem. A autorização de residência é processada pelo SEF/AIMA em Portugal, já o visto de residência é processado nos consulados portugueses. Ambos conferem direito de residência, mas têm procedimentos diferentes."
        }
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(legal_data, f, indent=2, ensure_ascii=False)
    
    print(f"Legal sample data created at {output_path}")
    return output_path

if __name__ == "__main__":
    # Example usage
    print("Creating legal sample data...")
    data_path = create_legal_sample_data()
    
    # Initialize trainer
    trainer = LegalPEFTTrainer(
        model_name="neuralmind/bert-base-portuguese-cased",
        use_peft=True,
        output_dir="./legal_fine_tuned_model"
    )
    
    # Prepare dataset
    train_dataset = trainer.prepare_dataset(data_path)
    
    # Train the model
    trainer.train(
        train_dataset=train_dataset,
        num_train_epochs=1,
        learning_rate=2e-4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        max_steps=10  # Small number for testing
    )
