#!/usr/bin/env python3
"""
Data preparation utilities for fine-tuning legal consultation models
"""

import json
import argparse
import os
import csv
import requests
import zipfile
import tarfile
import pandas as pd
from typing import List, Dict, Any
import urllib.parse
from pathlib import Path

def download_file(url: str, output_path: str):
    """Download a file from URL to local path"""
    print(f"Downloading {url}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"✓ Downloaded to {output_path}")
    return output_path

def extract_archive(archive_path: str, extract_to: str = None):
    """Extract zip, tar, or tar.gz files"""
    if extract_to is None:
        extract_to = os.path.dirname(archive_path)
    
    os.makedirs(extract_to, exist_ok=True)
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            extracted_files = zip_ref.namelist()
    elif archive_path.endswith(('.tar', '.tar.gz', '.tgz')):
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
            extracted_files = tar_ref.getnames()
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    
    print(f"✓ Extracted {len(extracted_files)} files to {extract_to}")
    return [os.path.join(extract_to, f) for f in extracted_files]

def convert_csv_to_legal_format(csv_file: str, output_file: str, 
                               question_col: str = "question", 
                               answer_col: str = "answer",
                               context_col: str = "context"):
    """Convert CSV file to legal consultation format"""
    df = pd.read_csv(csv_file)
    
    legal_data = []
    for _, row in df.iterrows():
        item = {
            "question": str(row.get(question_col, "")).strip(),
            "context": str(row.get(context_col, "")).strip() if context_col in df.columns else "",
            "answer": str(row.get(answer_col, "")).strip()
        }
        
        # Skip empty entries
        if item["question"] and item["answer"]:
            legal_data.append(item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(legal_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(legal_data)} entries from CSV to {output_file}")
    return output_file

def convert_jsonl_to_legal_format(jsonl_file: str, output_file: str):
    """Convert JSONL file to legal consultation format"""
    legal_data = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                
                # Try different field mappings
                question = (item.get("question") or item.get("input") or 
                           item.get("instruction") or item.get("query", "")).strip()
                
                answer = (item.get("answer") or item.get("output") or 
                         item.get("response") or item.get("text", "")).strip()
                
                context = (item.get("context") or item.get("background") or "").strip()
                
                if question and answer:
                    legal_data.append({
                        "question": question,
                        "context": context,
                        "answer": answer
                    })
                    
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON on line {line_num}")
                continue
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(legal_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(legal_data)} entries from JSONL to {output_file}")
    return output_file

def process_huggingface_dataset(dataset_name: str, output_file: str, subset: str = None):
    """Download and convert HuggingFace dataset to legal format"""
    try:
        from datasets import load_dataset
        
        print(f"Loading HuggingFace dataset: {dataset_name}")
        if subset:
            dataset = load_dataset(dataset_name, subset)
        else:
            dataset = load_dataset(dataset_name)
        
        # Use train split if available, otherwise first available split
        if 'train' in dataset:
            data = dataset['train']
        else:
            split_name = list(dataset.keys())[0]
            data = dataset[split_name]
            print(f"Using split: {split_name}")
        
        legal_data = []
        for item in data:
            # Try to map fields to our format
            question = ""
            answer = ""
            context = ""
            
            # Common field mappings
            if 'question' in item:
                question = item['question']
            elif 'input' in item:
                question = item['input']
            elif 'instruction' in item:
                question = item['instruction']
            elif 'text' in item and 'label' in item:
                question = item['text']
            
            if 'answer' in item:
                answer = item['answer']
            elif 'output' in item:
                answer = item['output']
            elif 'response' in item:
                answer = item['response']
            elif 'label' in item:
                answer = str(item['label'])
            
            if 'context' in item:
                context = item['context']
            
            if question and answer:
                legal_data.append({
                    "question": str(question).strip(),
                    "context": str(context).strip(),
                    "answer": str(answer).strip()
                })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(legal_data, f, indent=2, ensure_ascii=False)
        
        print(f"Converted {len(legal_data)} entries from HuggingFace dataset to {output_file}")
        return output_file
        
    except ImportError:
        print("Error: datasets library not installed. Run: pip install datasets")
        return None
    except Exception as e:
        print(f"Error processing HuggingFace dataset: {e}")
        return None

def setup_data_from_url(url: str, data_dir: str = "./data", output_name: str = "training_data.json"):
    """Complete pipeline: download, extract, and convert data from URL"""
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Parse URL to get filename
    parsed_url = urllib.parse.urlparse(url)
    filename = os.path.basename(parsed_url.path)
    if not filename:
        filename = "downloaded_data"
    
    # Download file
    download_path = os.path.join(data_dir, filename)
    downloaded_file = download_file(url, download_path)
    
    # Handle different file types
    output_path = os.path.join(data_dir, output_name)
    
    try:
        # Try to extract if it's an archive
        if filename.endswith(('.zip', '.tar', '.tar.gz', '.tgz')):
            extracted_files = extract_archive(downloaded_file, data_dir)
            
            # Look for data files in extracted content
            data_files = []
            for file_path in extracted_files:
                if file_path.endswith(('.json', '.jsonl', '.csv', '.txt')):
                    data_files.append(file_path)
            
            if not data_files:
                print("Warning: No recognized data files found in archive")
                return None
            
            # Use the first data file found
            main_data_file = data_files[0]
            print(f"Using data file: {main_data_file}")
            
        else:
            main_data_file = downloaded_file
        
        # Convert based on file extension
        if main_data_file.endswith('.csv'):
            return convert_csv_to_legal_format(main_data_file, output_path)
        elif main_data_file.endswith('.jsonl'):
            return convert_jsonl_to_legal_format(main_data_file, output_path)
        elif main_data_file.endswith('.json'):
            # Try to convert existing JSON format
            return convert_json_to_legal_format(main_data_file, output_path)
        else:
            print(f"Unsupported file format: {main_data_file}")
            return None
            
    except Exception as e:
        print(f"Error processing downloaded file: {e}")
        return None

def convert_json_to_legal_format(json_file: str, output_file: str):
    """Convert various JSON formats to legal consultation format"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    legal_data = []
    
    # Handle different JSON structures
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        # Look for common dataset keys
        if 'data' in data:
            items = data['data']
        elif 'examples' in data:
            items = data['examples']
        elif 'instances' in data:
            items = data['instances']
        else:
            # Treat as single item
            items = [data]
    else:
        print("Error: Unexpected JSON structure")
        return None
    
    for item in items:
        # Try multiple field mappings
        question_fields = ['question', 'input', 'instruction', 'query', 'pergunta']
        answer_fields = ['answer', 'output', 'response', 'resposta', 'text']
        context_fields = ['context', 'background', 'contexto', 'passage']
        
        question = ""
        answer = ""
        context = ""
        
        # Find question
        for field in question_fields:
            if field in item and item[field]:
                question = str(item[field]).strip()
                break
        
        # Find answer
        for field in answer_fields:
            if field in item and item[field]:
                answer = str(item[field]).strip()
                break
        
        # Find context
        for field in context_fields:
            if field in item and item[field]:
                context = str(item[field]).strip()
                break
        
        if question and answer:
            legal_data.append({
                "question": question,
                "context": context,
                "answer": answer
            })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(legal_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(legal_data)} entries from JSON to {output_file}")
    return output_file

def convert_alpaca_format(input_file: str, output_file: str):
    """Convert Alpaca-style dataset to our format"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted_data = []
    for item in data:
        converted_item = {
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "output": item.get("output", "")
        }
        converted_data.append(converted_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(converted_data)} examples from {input_file} to {output_file}")

def split_dataset(input_file: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Split dataset into train/validation/test sets"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # Save splits
    base_name = os.path.splitext(input_file)[0]
    
    with open(f"{base_name}_train.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(f"{base_name}_val.json", 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    with open(f"{base_name}_test.json", 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"Split {total} examples into:")
    print(f"  Train: {len(train_data)} examples ({base_name}_train.json)")
    print(f"  Val:   {len(val_data)} examples ({base_name}_val.json)")
    print(f"  Test:  {len(test_data)} examples ({base_name}_test.json)")

def validate_dataset(input_file: str):
    """Validate dataset format"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    errors = []
    warnings = []
    
    for i, item in enumerate(data):
        # Check required fields
        if "instruction" not in item:
            errors.append(f"Item {i}: Missing 'instruction' field")
        elif not item["instruction"].strip():
            warnings.append(f"Item {i}: Empty instruction")
        
        if "output" not in item:
            errors.append(f"Item {i}: Missing 'output' field")
        elif not item["output"].strip():
            warnings.append(f"Item {i}: Empty output")
        
        # Check optional fields
        if "input" not in item:
            item["input"] = ""
    
    print(f"Validation results for {input_file}:")
    print(f"  Total examples: {len(data)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Warnings: {len(warnings)}")
    
    if errors:
        print("\nErrors:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    
    if warnings:
        print("\nWarnings:")
        for warning in warnings[:5]:  # Show first 5 warnings
            print(f"  {warning}")
        if len(warnings) > 5:
            print(f"  ... and {len(warnings) - 5} more")
    
    return len(errors) == 0

def create_coding_dataset():
    """Create a sample coding instruction dataset"""
    coding_data = [
        {
            "instruction": "Write a Python function to calculate the factorial of a number",
            "input": "",
            "output": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)\n\n# Example usage:\n# print(factorial(5))  # Output: 120"
        },
        {
            "instruction": "Create a function to check if a string is a palindrome",
            "input": "",
            "output": "def is_palindrome(s):\n    # Remove spaces and convert to lowercase\n    s = s.replace(' ', '').lower()\n    # Check if string equals its reverse\n    return s == s[::-1]\n\n# Example usage:\n# print(is_palindrome('racecar'))  # Output: True"
        },
        {
            "instruction": "Write a Python function to find the maximum element in a list",
            "input": "",
            "output": "def find_max(lst):\n    if not lst:\n        return None\n    \n    max_val = lst[0]\n    for num in lst[1:]:\n        if num > max_val:\n            max_val = num\n    return max_val\n\n# Example usage:\n# print(find_max([1, 5, 3, 9, 2]))  # Output: 9"
        },
        {
            "instruction": "Implement a binary search algorithm",
            "input": "",
            "output": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    \n    while left <= right:\n        mid = (left + right) // 2\n        \n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    \n    return -1  # Target not found\n\n# Example usage:\n# arr = [1, 3, 5, 7, 9, 11]\n# print(binary_search(arr, 7))  # Output: 3"
        },
        {
            "instruction": "Create a class to represent a simple bank account",
            "input": "",
            "output": "class BankAccount:\n    def __init__(self, account_number, initial_balance=0):\n        self.account_number = account_number\n        self.balance = initial_balance\n    \n    def deposit(self, amount):\n        if amount > 0:\n            self.balance += amount\n            return True\n        return False\n    \n    def withdraw(self, amount):\n        if amount > 0 and amount <= self.balance:\n            self.balance -= amount\n            return True\n        return False\n    \n    def get_balance(self):\n        return self.balance\n\n# Example usage:\n# account = BankAccount('12345', 1000)\n# account.deposit(500)\n# print(account.get_balance())  # Output: 1500"
        }
    ]
    
    with open('coding_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(coding_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created coding dataset with {len(coding_data)} examples: coding_dataset.json")

def create_legal_immigration_dataset():
    """Create a comprehensive legal immigration dataset for Portugal"""
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
        },
        {
            "question": "Como solicitar visto D7 para aposentados?",
            "context": "Aposentado brasileiro com pensão mensal de €1500",
            "answer": "O visto D7 é destinado a aposentados e rendimentos passivos. Requisitos: 1) Comprovar rendimentos regulares (pensão, rendas, etc.) de pelo menos €760/mês (salário mínimo português); 2) Seguro de saúde válido em Portugal; 3) Alojamento em Portugal (contrato de arrendamento ou escritura); 4) Certificado de antecedentes criminais; 5) Declaração médica atestando capacidade física e mental. O processo demora 60-90 dias."
        },
        {
            "question": "Quais documentos preciso para visto de estudante?",
            "context": "Aceito em universidade portuguesa para mestrado",
            "answer": "Para visto de estudante em Portugal: 1) Carta de aceitação da instituição de ensino; 2) Comprovativo de meios de subsistência (€5148/ano ou €428/mês); 3) Seguro de saúde; 4) Alojamento em Portugal; 5) Certificado de antecedentes criminais; 6) Certificado médico; 7) Habilitações académicas reconhecidas. O visto permite trabalhar 20h/semana durante estudos."
        },
        {
            "question": "Como converter visto de turista em autorização de residência?",
            "context": "Estou em Portugal com visto de turista e encontrei emprego",
            "answer": "Não é possível converter visto de turista diretamente em autorização de residência para trabalho. Deve: 1) Sair de Portugal; 2) Solicitar visto de trabalho no consulado português no país de origem; 3) Aguardar aprovação; 4) Retornar a Portugal com visto adequado. Alternativamente, pode verificar se se enquadra nas manifestações de interesse para trabalho sazonal ou outras categorias especiais."
        },
        {
            "question": "Quanto custa o processo de naturalização portuguesa?",
            "context": "Interessado em adquirir nacionalidade portuguesa",
            "answer": "Os custos da naturalização portuguesa incluem: 1) Taxa de apreciação: €175; 2) Exame de português (se necessário): €85; 3) Certificados e documentos: €50-150; 4) Tradução juramentada de documentos estrangeiros: €20-40 por documento; 5) Taxa de emissão de certidão de nascimento portuguesa: €20. Total estimado: €350-450, podendo variar conforme a situação individual."
        },
        {
            "question": "Posso trabalhar em Portugal com visto D7?",
            "context": "Tenho visto D7 e gostaria de trabalhar",
            "answer": "O visto D7 não permite atividade profissional dependente (trabalho por conta de outrem). Permite apenas atividades independentes (trabalho por conta própria) desde que declaradas e autorizadas. Para trabalhar por conta de outrem, deve solicitar alteração da finalidade do título de residência junto ao AIMA, comprovando oferta de trabalho e cumprindo os requisitos legais."
        }
    ]
    
    with open('legal_immigration_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(legal_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created comprehensive legal immigration dataset with {len(legal_data)} examples: legal_immigration_dataset.json")

def convert_legal_format(input_file: str, output_file: str):
    """Convert different legal data formats to our Q&A format"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted_data = []
    for item in data:
        # Handle different input formats
        if "pergunta" in item and "resposta" in item:
            # Portuguese format
            converted_item = {
                "question": item.get("pergunta", ""),
                "context": item.get("contexto", ""),
                "answer": item.get("resposta", "")
            }
        elif "query" in item and "response" in item:
            # Generic Q&A format
            converted_item = {
                "question": item.get("query", ""),
                "context": item.get("context", ""),
                "answer": item.get("response", "")
            }
        else:
            # Our standard format
            converted_item = {
                "question": item.get("question", ""),
                "context": item.get("context", ""),
                "answer": item.get("answer", "")
            }
        converted_data.append(converted_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(converted_data)} legal examples from {input_file} to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Data preparation utilities for legal consultation")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Download and setup command
    download_parser = subparsers.add_parser('download', help='Download and setup data from URL')
    download_parser.add_argument('url', help='URL to download data from')
    download_parser.add_argument('--data_dir', default='./data', help='Directory to store data')
    download_parser.add_argument('--output_name', default='training_data.json', help='Output filename')
    
    # HuggingFace dataset command
    hf_parser = subparsers.add_parser('huggingface', help='Download HuggingFace dataset')
    hf_parser.add_argument('dataset_name', help='HuggingFace dataset name')
    hf_parser.add_argument('output_file', help='Output JSON file')
    hf_parser.add_argument('--subset', help='Dataset subset/config name')
    
    # Convert commands
    convert_parser = subparsers.add_parser('convert', help='Convert various formats to legal format')
    convert_subparsers = convert_parser.add_subparsers(dest='convert_type', help='Conversion type')
    
    # CSV conversion
    csv_parser = convert_subparsers.add_parser('csv', help='Convert CSV to legal format')
    csv_parser.add_argument('input_file', help='Input CSV file')
    csv_parser.add_argument('output_file', help='Output JSON file')
    csv_parser.add_argument('--question_col', default='question', help='Question column name')
    csv_parser.add_argument('--answer_col', default='answer', help='Answer column name')
    csv_parser.add_argument('--context_col', default='context', help='Context column name')
    
    # JSONL conversion
    jsonl_parser = convert_subparsers.add_parser('jsonl', help='Convert JSONL to legal format')
    jsonl_parser.add_argument('input_file', help='Input JSONL file')
    jsonl_parser.add_argument('output_file', help='Output JSON file')
    
    # JSON conversion
    json_parser = convert_subparsers.add_parser('json', help='Convert JSON to legal format')
    json_parser.add_argument('input_file', help='Input JSON file')
    json_parser.add_argument('output_file', help='Output JSON file')
    
    # Alpaca conversion (keeping original)
    alpaca_parser = convert_subparsers.add_parser('alpaca', help='Convert Alpaca format to legal format')
    alpaca_parser.add_argument('input_file', help='Input file in Alpaca format')
    alpaca_parser.add_argument('output_file', help='Output file in legal format')
    
    # Split command
    split_parser = subparsers.add_parser('split', help='Split dataset into train/val/test')
    split_parser.add_argument('input_file', help='Input JSON file')
    split_parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio')
    split_parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate dataset format')
    validate_parser.add_argument('input_file', help='Input JSON file to validate')
    
    # Create sample datasets
    subparsers.add_parser('create-coding', help='Create sample coding dataset')
    subparsers.add_parser('create-legal', help='Create legal immigration dataset')
    
    args = parser.parse_args()
    
    if args.command == 'download':
        result = setup_data_from_url(args.url, args.data_dir, args.output_name)
        if result:
            print(f"✓ Data ready for training at: {result}")
        else:
            print("❌ Failed to setup data")
    
    elif args.command == 'huggingface':
        result = process_huggingface_dataset(args.dataset_name, args.output_file, args.subset)
        if result:
            print(f"✓ HuggingFace dataset ready at: {result}")
    
    elif args.command == 'convert':
        if args.convert_type == 'csv':
            convert_csv_to_legal_format(args.input_file, args.output_file, 
                                      args.question_col, args.answer_col, args.context_col)
        elif args.convert_type == 'jsonl':
            convert_jsonl_to_legal_format(args.input_file, args.output_file)
        elif args.convert_type == 'json':
            convert_json_to_legal_format(args.input_file, args.output_file)
        elif args.convert_type == 'alpaca':
            convert_alpaca_format(args.input_file, args.output_file)
        else:
            convert_parser.print_help()
    
    elif args.command == 'split':
        split_dataset(args.input_file, args.train_ratio, args.val_ratio)
    elif args.command == 'validate':
        validate_dataset(args.input_file)
    elif args.command == 'create-coding':
        create_coding_dataset()
    elif args.command == 'create-legal':
        create_legal_immigration_dataset()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
