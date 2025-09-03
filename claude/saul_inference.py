#!/usr/bin/env python3
"""
Inference script for SaulLM Legal Question Answering model
"""

import torch
import argparse
import json
import logging
from model import SaulLMQuestionAnswering
from trainer import SaulLMTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SaulLMInference:
    """Inference class for SaulLM legal question answering"""
    
    def __init__(self, model_path: str = None, base_model_name: str = "pierreguillou/gpt2-small-portuguese"):
        """
        Initialize inference model.
        
        Args:
            model_path: Path to trained LoRA adapters (optional)
            base_model_name: Base model name
        """
        self.model_path = model_path
        self.base_model_name = base_model_name
        
        if model_path and os.path.exists(model_path):
            # Load trained model
            logger.info(f"Loading trained model from {model_path}")
            self.model = SaulLMQuestionAnswering.load_trained_model(model_path, base_model_name)
        else:
            # Load base model
            logger.info(f"Loading base model: {base_model_name}")
            self.model = SaulLMQuestionAnswering(
                model_name=base_model_name,
                use_quantization=True,
                use_peft=False
            )
    
    def generate_answer(self, question: str, context: str = "", **kwargs) -> str:
        """Generate answer for a legal question"""
        return self.model.generate_answer(question, context, **kwargs)
    
    def batch_inference(self, questions: list, contexts: list = None, **kwargs) -> list:
        """Run inference on a batch of questions"""
        if contexts is None:
            contexts = [""] * len(questions)
        
        answers = []
        for i, question in enumerate(questions):
            context = contexts[i] if i < len(contexts) else ""
            answer = self.generate_answer(question, context, **kwargs)
            answers.append(answer)
        
        return answers
    
    def interactive_consultation(self):
        """Interactive legal consultation mode"""
        print("🏛️  SaulLM - Consulta Jurídica de Imigração Portugal 🇵🇹")
        print("=" * 60)
        print("Assistente legal baseado em IA para questões de imigração em Portugal.")
        print("Digite 'sair' para terminar, 'help' para ajuda.")
        print("-" * 60)
        
        while True:
            question = input("\n💼 Sua pergunta jurídica: ").strip()
            
            if question.lower() in ['sair', 'exit', 'quit', 'q']:
                print("Obrigado por usar o assistente jurídico SaulLM. Até breve!")
                break
            
            if question.lower() in ['help', 'ajuda', 'h']:
                self._show_help()
                continue
            
            if not question:
                print("Por favor, faça uma pergunta.")
                continue
            
            context = input("📋 Contexto adicional (opcional): ").strip()
            
            print("\n⚖️  Analisando sua consulta com SaulLM...")
            try:
                answer = self.generate_answer(
                    question, 
                    context, 
                    max_new_tokens=300,
                    temperature=0.3,
                    top_p=0.9
                )
                
                print(f"\n📝 Resposta Legal (SaulLM):")
                print("-" * 40)
                print(answer)
                print("-" * 40)
                print("\n⚠️  AVISO: Esta resposta é gerada por IA e é apenas informativa.")
                print("Para questões específicas, consulte sempre um advogado especializado.")
                
            except Exception as e:
                print(f"❌ Erro ao gerar resposta: {e}")
                print("Por favor, tente reformular sua pergunta.")
    
    def _show_help(self):
        """Show help information"""
        print("\n📚 Ajuda - SaulLM Legal Assistant:")
        print("• Faça perguntas sobre imigração em Portugal")
        print("• Seja específico em suas perguntas")
        print("• Forneça contexto quando necessário")
        print("\nExemplos de perguntas:")
        print("- Como obter visto de trabalho para Portugal?")
        print("- Quanto tempo demora para conseguir cidadania portuguesa?")
        print("- Posso trazer minha família para Portugal?")
        print("- Que documentos preciso apostilar no Brasil?")

def main():
    parser = argparse.ArgumentParser(description="SaulLM Legal Question Answering Inference")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to trained LoRA adapters")
    parser.add_argument("--base_model", type=str, default="pierreguillou/gpt2-small-portuguese",
                       help="Base model name (default: Portuguese GPT-2 small)")
    parser.add_argument("--question", type=str, default=None,
                       help="Single legal question to process")
    parser.add_argument("--context", type=str, default="",
                       help="Context for the legal question")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive consultation mode")
    parser.add_argument("--batch_file", type=str, default=None,
                       help="JSON file with batch questions")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file for batch results")
    parser.add_argument("--max_tokens", type=int, default=300,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.3,
                       help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling parameter")
    
    args = parser.parse_args()
    
    # Initialize inference model
    logger.info("Initializing SaulLM inference model...")
    inference = SaulLMInference(args.model_path, args.base_model)
    
    if args.interactive:
        # Interactive mode
        inference.interactive_consultation()
        
    elif args.batch_file:
        # Batch processing mode
        logger.info(f"Processing batch file: {args.batch_file}")
        
        with open(args.batch_file, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
        
        questions = [item['question'] for item in batch_data]
        contexts = [item.get('context', '') for item in batch_data]
        
        answers = inference.batch_inference(
            questions, 
            contexts,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        # Prepare results
        results = []
        for i, item in enumerate(batch_data):
            result = {
                'question': item['question'],
                'context': item.get('context', ''),
                'generated_answer': answers[i],
                'original_answer': item.get('answer', '')
            }
            results.append(result)
        
        # Save results
        output_file = args.output_file or 'batch_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Batch processing completed. Results saved to {output_file}")
        
    elif args.question:
        # Single question mode
        logger.info("Processing single question...")
        answer = inference.generate_answer(
            args.question,
            args.context,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        print(f"Pergunta: {args.question}")
        if args.context:
            print(f"Contexto: {args.context}")
        print(f"\nResposta Legal (SaulLM): {answer}")
        print("\n⚠️  AVISO: Esta resposta é gerada por IA e é apenas informativa.")
        print("Para questões específicas, consulte sempre um advogado especializado.")
        
    else:
        print("Use --question para consulta única, --interactive para modo interativo,")
        print("ou --batch_file para processamento em lote.")
        print("\nExemplos:")
        print("  python saul_inference.py --interactive")
        print("  python saul_inference.py --question 'Como obter visto de trabalho?'")
        print("  python saul_inference.py --model_path ./legal_saul_model --interactive")

if __name__ == "__main__":
    import os
    main()
