#!/usr/bin/env python3
"""
Improved inference script using a Portuguese generative model
"""

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import json

class PortugueseLegalConsultant:
    def __init__(self, base_model_name: str = "microsoft/DialoGPT-medium"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use a Portuguese model or multilingual model that supports generation
        # Since Portuguese DialoGPT might not be available, we'll use a workaround
        print(f"Loading model: {base_model_name}")
        
        try:
            # Try Portuguese-specific models first
            models_to_try = [
                "pierreguillou/gpt2-small-portuguese",  # Portuguese GPT-2
                "microsoft/DialoGPT-medium",            # Multilingual dialog model
                "gpt2"                                  # Fallback to base GPT-2
            ]
            
            self.model = None
            self.tokenizer = None
            
            for model_name in models_to_try:
                try:
                    print(f"Trying to load {model_name}...")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModelForCausalLM.from_pretrained(model_name)
                    
                    # Set pad token if not available
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                        
                    print(f"✅ Successfully loaded {model_name}")
                    break
                except Exception as e:
                    print(f"❌ Failed to load {model_name}: {e}")
                    continue
            
            if self.model is None:
                raise Exception("No suitable model could be loaded")
                
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using pipeline as fallback...")
            self.use_pipeline = True
            self.generator = pipeline("text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1)
    
    def generate_legal_advice(
        self, 
        question: str, 
        context: str = "", 
        max_length: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        """Generate legal advice for immigration questions"""
        
        # Create a legal prompt template
        if context:
            prompt = f"""Pergunta jurídica sobre imigração em Portugal:
Pergunta: {question}
Contexto: {context}
Resposta legal detalhada: """
        else:
            prompt = f"""Pergunta jurídica sobre imigração em Portugal:
Pergunta: {question}
Resposta legal detalhada: """
        
        try:
            if hasattr(self, 'use_pipeline') and self.use_pipeline:
                # Use pipeline
                result = self.generator(
                    prompt,
                    max_length=len(prompt.split()) + max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=50256,  # GPT-2 pad token
                    num_return_sequences=1
                )
                response = result[0]['generated_text']
                # Extract just the generated part
                response = response[len(prompt):].strip()
            else:
                # Use model directly
                inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1
                    )
                
                # Decode response
                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract just the generated part
                response = full_response[len(prompt):].strip()
            
            # Clean up the response
            if response:
                # Remove any incomplete sentences at the end
                sentences = response.split('.')
                if len(sentences) > 1 and sentences[-1].strip() == "":
                    sentences = sentences[:-1]
                response = '. '.join(sentences)
                if response and not response.endswith('.'):
                    response += '.'
            else:
                response = "Desculpe, não consigo gerar uma resposta adequada para esta pergunta legal específica."
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Ocorreu um erro ao gerar a resposta. Por favor, tente reformular sua pergunta."
    
    def interactive_consultation(self):
        """Interactive legal consultation mode"""
        print("🏛️  Consulta Jurídica - Imigração Portugal 🇵🇹")
        print("=" * 60)
        print("Bem-vindo ao assistente de consulta jurídica para assuntos de imigração em Portugal.")
        print("Digite 'sair' para terminar a consulta.")
        print("⚠️  AVISO: Respostas são apenas informativas e não substituem consulta a advogado.")
        print("-" * 60)
        
        while True:
            question = input("\n💼 Sua pergunta jurídica: ").strip()
            if question.lower() in ['sair', 'exit', 'quit', 'q']:
                print("Obrigado por usar o assistente jurídico. Até breve!")
                break
            
            if not question:
                print("Por favor, faça uma pergunta.")
                continue
            
            context = input("📋 Contexto adicional (opcional): ").strip()
            
            print("\n⚖️  Analisando sua consulta...")
            advice = self.generate_legal_advice(question, context)
            
            print(f"\n📝 Resposta Legal:")
            print("-" * 40)
            print(advice)
            print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description="Legal Consultation Assistant for Portugal Immigration")
    parser.add_argument("--question", type=str, default=None,
                       help="Single legal question to process")
    parser.add_argument("--context", type=str, default="",
                       help="Context for the legal question")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive consultation mode")
    parser.add_argument("--max_length", type=int, default=150,
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.3,
                       help="Generation temperature (lower = more conservative)")
    
    args = parser.parse_args()
    
    # Initialize consultant
    print("Carregando assistente de consulta jurídica...")
    consultant = PortugueseLegalConsultant()
    
    if args.interactive:
        # Interactive mode
        consultant.interactive_consultation()
    elif args.question:
        # Single question mode
        advice = consultant.generate_legal_advice(
            args.question,
            args.context,
            args.max_length,
            args.temperature
        )
        print(f"Pergunta: {args.question}")
        if args.context:
            print(f"Contexto: {args.context}")
        print(f"\nResposta Legal: {advice}")
        print("\n⚠️  AVISO: Esta resposta é apenas informativa e não substitui a consulta a um advogado.")
    else:
        print("Use --question para consulta única ou --interactive para modo interativo")
        print("Exemplo: python legal_assistant.py --question 'Como obter visto de trabalho?'")

if __name__ == "__main__":
    main()
