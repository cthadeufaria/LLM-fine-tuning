#!/usr/bin/env python3
"""
Practical Legal Assistant using rule-based responses and templates
"""

import argparse
import json
import re
from typing import Dict, List, Tuple

class PracticalLegalAssistant:
    def __init__(self):
        """Initialize the legal assistant with knowledge base"""
        self.knowledge_base = {
            "visto_trabalho": {
                "keywords": ["visto", "trabalho", "trabalhar", "emprego", "job", "work"],
                "response": """Para obter um visto de trabalho em Portugal sendo brasileiro, você precisa:

1. **Oferta de Emprego**: Ter uma oferta de trabalho de uma empresa portuguesa
2. **Contrato de Trabalho**: Assinar contrato com duração mínima definida
3. **Documentação Necessária**:
   - Passaporte brasileiro válido
   - Certificado de antecedentes criminais
   - Comprovante de qualificações/diplomas
   - Seguro de saúde
   - Comprovante de alojamento

4. **Processo**:
   - Solicitar no consulado português no Brasil
   - Prazo: 15-60 dias úteis
   - Taxa consular aplicável

5. **Após Aprovação**:
   - Recolher visto no consulado
   - Entrar em Portugal em até 90 dias
   - Solicitar autorização de residência no SEF"""
            },
            
            "residencia": {
                "keywords": ["residência", "residencia", "morar", "viver", "permanência"],
                "response": """Para obter autorização de residência em Portugal:

1. **Tipos de Autorização**:
   - Autorização de Residência Temporária (1-2 anos)
   - Autorização de Residência Permanente (após 5 anos)

2. **Requisitos Básicos**:
   - Entrada legal no país
   - Meios de subsistência comprovados
   - Seguro de saúde
   - Alojamento adequado
   - Registo criminal limpo

3. **Documentação**:
   - Formulário de pedido preenchido
   - Fotografias tipo passe
   - Passaporte e visto válidos
   - Contrato de trabalho ou comprovativo de rendimentos

4. **Processo no SEF**:
   - Agendamento obrigatório
   - Prazo máximo: 90 dias após entrada
   - Taxa: aproximadamente 75€"""
            },
            
            "cidadania": {
                "keywords": ["cidadania", "nacionalidade", "português", "passaporte"],
                "response": """Para adquirir cidadania portuguesa sendo brasileiro:

1. **Vias Principais**:
   - Por naturalização (após 5 anos de residência)
   - Por ascendência (pais/avós portugueses)
   - Por casamento (3 anos de casamento + 1 ano de residência)

2. **Naturalização - Requisitos**:
   - 5 anos de residência legal ininterrupta
   - Conhecimentos básicos de português (A2)
   - Ligação efetiva ao território nacional
   - Idoneidade civil e criminal

3. **Documentação Necessária**:
   - Certidão de nascimento apostilada
   - Certificados criminais (Brasil e Portugal)
   - Comprovativo de conhecimento da língua
   - Comprovativo de rendimentos
   - Registo biográfico

4. **Processo**:
   - Requerimento na Conservatória dos Registos Centrais
   - Prazo: 12-24 meses
   - Taxa: cerca de 200€"""
            },
            
            "documentos": {
                "keywords": ["documentos", "papéis", "certidão", "apostila", "legalização"],
                "response": """Documentos essenciais para imigração portuguesa:

1. **Documentos Brasileiros** (com apostila de Haia):
   - Certidão de nascimento
   - Certidão de casamento (se aplicável)
   - Diploma de graduação/certificados
   - Antecedentes criminais da Polícia Federal

2. **Documentos Portugueses**:
   - Registo criminal português (após residência)
   - Comprovativo de rendimentos
   - Comprovativo de alojamento
   - Seguro de saúde

3. **Procedimentos**:
   - Apostila de Haia no Brasil (cartórios)
   - Tradução juramentada em Portugal (se necessário)
   - Validação de diplomas no NARIC

4. **Prazos de Validade**:
   - Antecedentes criminais: 3 meses
   - Outros documentos: geralmente 6 meses"""
            },
            
            "reagrupamento": {
                "keywords": ["família", "cônjuge", "filhos", "reagrupamento", "reunião"],
                "response": """Reagrupamento familiar em Portugal:

1. **Quem Pode Solicitar**:
   - Residentes legais em Portugal há mais de 1 ano
   - Refugiados e beneficiários de proteção subsidiária

2. **Familiares Elegíveis**:
   - Cônjuge ou parceiro em união de facto
   - Filhos menores ou maiores dependentes
   - Pais ou avós dependentes

3. **Requisitos**:
   - Alojamento adequado para a família
   - Recursos financeiros suficientes
   - Seguro de saúde para todos
   - Registo criminal limpo

4. **Processo**:
   - Pedido no consulado português no país de origem dos familiares
   - Prazo: 60-90 dias
   - Após chegada: pedido de autorização de residência no SEF"""
            }
        }
        
        # Load training data if available for additional context
        try:
            with open("legal_sample_training_data.json", "r", encoding="utf-8") as f:
                self.training_data = json.load(f)
        except:
            self.training_data = []
    
    def find_best_match(self, question: str) -> Tuple[str, str]:
        """Find the best matching response for a question"""
        question_lower = question.lower()
        
        # Score each category based on keyword matches
        scores = {}
        for category, data in self.knowledge_base.items():
            score = 0
            for keyword in data["keywords"]:
                if keyword in question_lower:
                    score += 1
            scores[category] = score
        
        # Find the category with highest score
        if scores and max(scores.values()) > 0:
            best_category = max(scores, key=scores.get)
            return best_category, self.knowledge_base[best_category]["response"]
        
        # If no good match, provide general guidance
        return "geral", self.get_general_response()
    
    def get_general_response(self) -> str:
        """Provide general guidance when no specific match is found"""
        return """Para questões de imigração em Portugal, recomendo:

1. **Consulte o SEF** (Serviço de Estrangeiros e Fronteiras):
   - Website: www.sef.pt
   - Linha de apoio: 808 202 653

2. **Consulados Portugueses no Brasil**:
   - Para vistos e informações preliminares

3. **Advogado Especializado**:
   - Para casos complexos ou específicos

4. **Documentação Sempre Atualizada**:
   - Mantenha todos os documentos em dia
   - Verifique prazos de validade

5. **Recursos Oficiais**:
   - Portal ePortugal: eportugal.gov.pt
   - Guia do Imigrante: www.acm.gov.pt

Para perguntas específicas sobre vistos de trabalho, residência, cidadania ou reagrupamento familiar, formule sua pergunta com mais detalhes."""
    
    def enhance_with_context(self, response: str, context: str) -> str:
        """Enhance response with user-provided context"""
        if not context:
            return response
        
        enhanced = f"{response}\n\n**Considerando seu contexto específico** ({context}):\n"
        
        # Add context-specific advice
        context_lower = context.lower()
        if "engenheiro" in context_lower or "software" in context_lower:
            enhanced += "- Como profissional de TI, você pode se beneficiar do regime especial para profissões em falta\n"
            enhanced += "- Considere certificar suas competências tecnológicas em Portugal\n"
        
        if any(word in context_lower for word in ["anos", "experiência", "experience"]):
            enhanced += "- Sua experiência profissional é um ponto forte no processo\n"
            enhanced += "- Prepare documentos que comprovem sua experiência e qualificações\n"
        
        if "casado" in context_lower or "married" in context_lower:
            enhanced += "- Se seu cônjuge for português ou residente legal, pode facilitar o processo\n"
            enhanced += "- Considere o reagrupamento familiar se aplicável\n"
        
        return enhanced
    
    def generate_legal_advice(self, question: str, context: str = "") -> str:
        """Generate legal advice based on question and context"""
        category, response = self.find_best_match(question)
        
        # Enhance response with context
        if context:
            response = self.enhance_with_context(response, context)
        
        # Add disclaimer
        response += "\n\n**⚠️ IMPORTANTE**: Esta informação é apenas orientativa. Para casos específicos, consulte sempre um advogado especializado em direito de imigração ou contacte diretamente o SEF."
        
        return response
    
    def interactive_consultation(self):
        """Interactive legal consultation mode"""
        print("🏛️  Consulta Jurídica - Imigração Portugal 🇵🇹")
        print("=" * 60)
        print("Assistente jurídico para assuntos de imigração em Portugal.")
        print("Digite 'sair' para terminar, 'help' para ver tópicos disponíveis.")
        print("-" * 60)
        
        while True:
            question = input("\n💼 Sua pergunta jurídica: ").strip()
            
            if question.lower() in ['sair', 'exit', 'quit', 'q']:
                print("Obrigado por usar o assistente jurídico. Até breve!")
                break
            
            if question.lower() in ['help', 'ajuda', 'h']:
                self.show_help()
                continue
            
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
    
    def show_help(self):
        """Show available topics"""
        print("\n📚 Tópicos Disponíveis:")
        print("• Visto de trabalho")
        print("• Autorização de residência")
        print("• Cidadania portuguesa")
        print("• Documentos necessários")
        print("• Reagrupamento familiar")
        print("\nExemplos de perguntas:")
        print("- Como obter visto de trabalho para Portugal?")
        print("- Que documentos preciso para residência?")
        print("- Como adquirir cidadania portuguesa?")

def main():
    parser = argparse.ArgumentParser(description="Assistente Jurídico - Imigração Portugal")
    parser.add_argument("--question", type=str, default=None,
                       help="Pergunta jurídica específica")
    parser.add_argument("--context", type=str, default="",
                       help="Contexto adicional para a pergunta")
    parser.add_argument("--interactive", action="store_true",
                       help="Modo interativo de consulta")
    
    args = parser.parse_args()
    
    # Initialize assistant
    assistant = PracticalLegalAssistant()
    
    if args.interactive:
        assistant.interactive_consultation()
    elif args.question:
        advice = assistant.generate_legal_advice(args.question, args.context)
        print(f"Pergunta: {args.question}")
        if args.context:
            print(f"Contexto: {args.context}")
        print(f"\nResposta Legal:\n{advice}")
    else:
        print("Use --question para consulta única ou --interactive para modo interativo")
        print("Exemplo: python practical_assistant.py --question 'Como obter visto de trabalho?'")

if __name__ == "__main__":
    main()
