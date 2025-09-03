#!/usr/bin/env python3
"""
Main script for training SaulLM Question Answering model for Legal Consultation
Uses quantized SaulLM with LoRA adapters for Portuguese immigration law
"""

import argparse
import json
import os
import torch
import logging
from trainer import SaulLMTrainer, train_legal_qa_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(output_path: str = "legal_sample_training_data.json"):
    """Create sample legal QA data if none exists"""
    sample_data = [
        {
            "question": "Como posso obter um visto de trabalho para Portugal sendo brasileiro?",
            "context": "Sou engenheiro de software com 5 anos de experiência",
            "answer": "Para obter um visto de trabalho em Portugal sendo brasileiro, você precisa: 1) Ter uma oferta de emprego de empresa portuguesa, 2) Apresentar contrato de trabalho, 3) Reunir documentação necessária (passaporte, antecedentes criminais, qualificações, seguro saúde, comprovante de alojamento), 4) Solicitar no consulado português no Brasil (prazo 15-60 dias úteis), 5) Após aprovação, recolher visto e entrar em Portugal em até 90 dias para solicitar autorização de residência no SEF. Como engenheiro de software, você pode se beneficiar do regime especial para profissões em falta."
        },
        {
            "question": "Quanto tempo demora para obter cidadania portuguesa?",
            "context": "Moro em Portugal há 4 anos com autorização de residência",
            "answer": "Para obter cidadania portuguesa por naturalização, você precisa de 5 anos de residência legal ininterrupta. Como você já tem 4 anos, precisará esperar mais 1 ano. Além disso, deve comprovar: conhecimentos básicos de português (nível A2), ligação efetiva ao território nacional, idoneidade civil e criminal. O processo demora entre 12-24 meses após o requerimento na Conservatória dos Registos Centrais, com taxa de cerca de 200€."
        },
        {
            "question": "Posso trazer minha família para Portugal?",
            "context": "Tenho autorização de residência e sou casado com dois filhos menores",
            "answer": "Sim, você pode solicitar reagrupamento familiar. Como residente legal em Portugal, pode trazer: cônjuge, filhos menores ou maiores dependentes. Requisitos: alojamento adequado, recursos financeiros suficientes, seguro de saúde para todos, registo criminal limpo. O pedido é feito no consulado português no país de origem dos familiares, com prazo de 60-90 dias. Após chegada, deve solicitar autorização de residência no SEF para os familiares."
        },
        {
            "question": "Que documentos preciso apostilar no Brasil?",
            "context": "Vou solicitar visto de trabalho",
            "answer": "Para visto de trabalho, deve apostilar (Apostila de Haia): certidão de nascimento, certidão de casamento (se casado), diploma de graduação/certificados profissionais, antecedentes criminais da Polícia Federal. A apostila deve ser feita em cartórios autorizados no Brasil. Documentos apostilados têm validade de 6 meses (antecedentes criminais: 3 meses). Em Portugal, pode precisar de tradução juramentada se os documentos não estiverem em português."
        },
        {
            "question": "Como renovar minha autorização de residência?",
            "context": "Minha autorização de residência vence em 2 meses",
            "answer": "Deve renovar sua autorização de residência antes do vencimento. Processo: 1) Agendar atendimento no SEF com 30-60 dias de antecedência, 2) Reunir documentação (formulário preenchido, fotografias, passaporte, comprovativo de rendimentos, seguro saúde, comprovativo de alojamento, registo criminal), 3) Comparecer ao atendimento agendado, 4) Pagar taxa de renovação (cerca de 75€). Se não conseguir agendar antes do vencimento, pode solicitar prorrogação de permanência."
        }
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Sample data created at {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Train SaulLM for Legal Question Answering")
    parser.add_argument("--model_name", type=str, default="mradermacher/SaulLM-54B-Instruct-i1-GGUF", 
                       help="Base model name. Options: mradermacher/SaulLM-54B-Instruct-i1-GGUF (default, quantized GGUF), pierreguillou/gpt2-small-portuguese (124M), microsoft/DialoGPT-medium (345M), Equall/Saul-7B-Instruct-v1 (7B)")
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to legal QA data JSON file")
    parser.add_argument("--output_dir", type=str, default="./legal_qa_model", 
                       help="Output directory for the fine-tuned model")
    parser.add_argument("--max_length", type=int, default=512, 
                       help="Maximum sequence length (512 for small models, 1024+ for larger)")
    parser.add_argument("--num_epochs", type=int, default=3, 
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, 
                       help="Learning rate for LoRA training")
    parser.add_argument("--batch_size", type=int, default=2, 
                       help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, 
                       help="Gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=50, 
                       help="Number of warmup steps")
    parser.add_argument("--save_steps", type=int, default=100, 
                       help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=5, 
                       help="Log every N steps")
    parser.add_argument("--eval_steps", type=int, default=100, 
                       help="Evaluation steps")
    parser.add_argument("--use_quantization", action="store_true", default=False,
                       help="Use 4-bit quantization")
    parser.add_argument("--no_quantization", action="store_true",
                       help="Disable quantization")
    parser.add_argument("--create_sample_data", action="store_true",
                       help="Create sample legal QA data")
    parser.add_argument("--eval_data_path", type=str, default=None,
                       help="Path to evaluation data")
    parser.add_argument("--test_generation", action="store_true",
                       help="Test model generation after training")
    args = parser.parse_args()
    
    # Handle quantization setting
    use_quantization = args.use_quantization and not args.no_quantization
    
    # Handle data path
    if args.data_path is None and not args.create_sample_data:
        logger.error("No data path provided!")
        logger.info("Options:")
        logger.info("1. Create sample data: python main.py --create_sample_data")
        logger.info("2. Use your own data: python main.py --data_path your_data.json")
        logger.info("3. Setup data interactively: python setup_data.py --interactive")
        return
    
    # Create sample data if requested
    if args.create_sample_data:
        logger.info("Creating sample legal QA data...")
        sample_path = create_sample_data("legal_sample_training_data.json")
        if args.data_path is None:
            args.data_path = sample_path
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        logger.error(f"Data file {args.data_path} not found!")
        logger.info("Use --create_sample_data to create sample data, or provide a valid data path.")
        return
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.info("CUDA not available, using CPU")
        if use_quantization:
            logger.warning("Quantization may not work on CPU, consider using --no_quantization")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Train the model using the convenience function
        logger.info("Starting SaulLM legal QA training...")
        trainer = train_legal_qa_model(
            data_path=args.data_path,
            output_dir=args.output_dir,
            model_name=args.model_name,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_quantization=use_quantization,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            max_length=args.max_length,
            eval_data_path=args.eval_data_path,
        )
        
        logger.info(f"Training completed! Model saved to {args.output_dir}")
        
        # Test generation if requested
        if args.test_generation:
            logger.info("Testing model generation...")
            test_questions = [
                "Como obter visto de trabalho para Portugal?",
                "Quanto tempo demora para obter cidadania portuguesa?",
                "Posso trazer minha família para Portugal?"
            ]
            
            for question in test_questions:
                logger.info(f"\nTesting: {question}")
                answer = trainer.generate_answer(question, max_new_tokens=200, temperature=0.7)
                logger.info(f"Answer: {answer[:200]}...")
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise e

if __name__ == "__main__":
    main()
