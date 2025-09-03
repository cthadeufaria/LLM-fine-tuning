#!/usr/bin/env python3
"""
Inference script for fine-tuned Legal BERT Seq2Seq model
"""

import torch
import argparse
from transformers import BertTokenizer, EncoderDecoderModel
from peft import PeftModel
import json

class LegalConsultationInference:
    def __init__(self, base_model_name: str, peft_model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(base_model_name)
        
        # Add special tokens for legal consultation
        special_tokens = {
            "additional_special_tokens": [
                "[LEGAL_Q]", "[LEGAL_A]", "[IMMIGRATION]", "[PORTUGAL]", 
                "[VISA]", "[RESIDENCY]", "[CITIZENSHIP]", "[DOCUMENT]"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        
        # Load base model
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            base_model_name, base_model_name
        )
        
        # Resize token embeddings to accommodate new tokens
        self.model.encoder.resize_token_embeddings(len(self.tokenizer))
        self.model.decoder.resize_token_embeddings(len(self.tokenizer))
        
        # Configure the model for generation
        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.model.config.eos_token_id = self.tokenizer.sep_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Load PEFT adapters if provided
        if peft_model_path:
            print(f"Loading PEFT adapters from {peft_model_path}")
            self.model = PeftModel.from_pretrained(self.model, peft_model_path)
        
        self.model.to(self.device)
        self.model.eval()
        print(f"Legal consultation model loaded on {self.device}")
    
    def generate_legal_advice(
        self, 
        question: str, 
        context: str = "", 
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ):
        """Generate legal advice for given question and context"""
        
        # Format input with legal tokens
        if context:
            input_text = f"[LEGAL_Q] {question} [IMMIGRATION] {context}"
        else:
            input_text = f"[LEGAL_Q] {question}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length//2,  # Reserve space for output
            padding=True
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.sep_token_id,
                num_return_sequences=1,
                early_stopping=True,
                num_beams=3 if not do_sample else 1
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the answer part (remove input echo)
        if "[LEGAL_A]" in response:
            response = response.split("[LEGAL_A]")[-1].strip()
        elif input_text in response:
            response = response.replace(input_text, "").strip()
        
        return response
    
    def interactive_legal_consultation(self):
        """Interactive legal consultation mode"""
        print("🏛️  Consulta Jurídica - Imigração Portugal 🇵🇹")
        print("=" * 60)
        print("Bem-vindo ao assistente de consulta jurídica para assuntos de imigração em Portugal.")
        print("Digite 'sair' para terminar a consulta.")
        print("-" * 60)
        
        while True:
            question = input("\n💼 Sua pergunta jurídica: ").strip()
            if question.lower() in ['sair', 'exit', 'quit', 'q']:
                print("Obrigado por usar o assistente jurídico. Até breve!")
                break
            
            context = input("📋 Contexto adicional (opcional): ").strip()
            
            print("\n⚖️  Analisando sua consulta...")
            advice = self.generate_legal_advice(question, context)
            
            print(f"\n📝 Resposta Legal:")
            print("-" * 40)
            print(advice)
            print("-" * 40)
            print("\n⚠️  AVISO: Esta resposta é apenas informativa e não substitui a consulta a um advogado.")

def main():
    parser = argparse.ArgumentParser(description="Legal Consultation with fine-tuned BERT")
    parser.add_argument("--base_model", type=str, default="neuralmind/bert-base-portuguese-cased",
                       help="Base BERT model name")
    parser.add_argument("--peft_model", type=str, default=None,
                       help="Path to PEFT model adapters")
    parser.add_argument("--question", type=str, default=None,
                       help="Single legal question to process")
    parser.add_argument("--context", type=str, default="",
                       help="Context for the legal question")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive consultation mode")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.3,
                       help="Generation temperature (lower = more conservative)")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling")
    
    args = parser.parse_args()
    
    # Initialize model
    print("Carregando modelo de consulta jurídica...")
    model = LegalConsultationInference(args.base_model, args.peft_model)
    
    if args.interactive:
        # Interactive mode
        model.interactive_legal_consultation()
    elif args.question:
        # Single question mode
        advice = model.generate_legal_advice(
            args.question,
            args.context,
            args.max_length,
            args.temperature,
            args.top_p
        )
        print(f"Pergunta: {args.question}")
        if args.context:
            print(f"Contexto: {args.context}")
        print(f"\nResposta Legal: {advice}")
        print("\n⚠️  AVISO: Esta resposta é apenas informativa e não substitui a consulta a um advogado.")
    else:
        print("Por favor, forneça --question para consulta única ou use --interactive para modo interativo")

if __name__ == "__main__":
    main()
