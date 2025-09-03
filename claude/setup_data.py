#!/usr/bin/env python3
"""
Comprehensive data setup script for legal consultation fine-tuning
"""

import argparse
import os
import json
from data_utils import setup_data_from_url, process_huggingface_dataset, validate_dataset

def setup_legal_data():
    """Interactive setup for legal consultation data"""
    print("🏛️  Legal Consultation Data Setup")
    print("=" * 50)
    
    # Common legal datasets and sources
    suggested_sources = {
        "1": {
            "name": "Portuguese Legal Q&A Dataset (Example)",
            "url": "https://example.com/legal_qa.json",
            "description": "Pre-processed Portuguese legal consultation dataset"
        },
        "2": {
            "name": "HuggingFace Legal Dataset",
            "dataset": "legal_data_portuguese",
            "description": "Legal dataset from HuggingFace Hub"
        },
        "3": {
            "name": "Custom URL",
            "description": "Download from your own URL"
        },
        "4": {
            "name": "Local File",
            "description": "Convert local file to legal format"
        }
    }
    
    print("Available data sources:")
    for key, source in suggested_sources.items():
        print(f"{key}. {source['name']}")
        print(f"   {source['description']}")
        print()
    
    choice = input("Select data source (1-4): ").strip()
    
    if choice == "1":
        # Example dataset - you would replace with real URL
        print("Note: This is an example URL. Replace with your actual data source.")
        url = input("Enter the actual URL for your legal dataset: ").strip()
        if url:
            return setup_data_from_url(url, "./data", "legal_training_data.json")
    
    elif choice == "2":
        # HuggingFace dataset
        dataset_name = input("Enter HuggingFace dataset name: ").strip()
        if dataset_name:
            subset = input("Enter subset name (optional, press Enter to skip): ").strip() or None
            return process_huggingface_dataset(dataset_name, "./data/legal_training_data.json", subset)
    
    elif choice == "3":
        # Custom URL
        url = input("Enter URL to download data from: ").strip()
        if url:
            return setup_data_from_url(url, "./data", "legal_training_data.json")
    
    elif choice == "4":
        # Local file
        file_path = input("Enter path to your local file: ").strip()
        if os.path.exists(file_path):
            from data_utils import convert_json_to_legal_format, convert_csv_to_legal_format, convert_jsonl_to_legal_format
            
            output_file = "./data/legal_training_data.json"
            os.makedirs("./data", exist_ok=True)
            
            if file_path.endswith('.csv'):
                question_col = input("Question column name (default: question): ").strip() or "question"
                answer_col = input("Answer column name (default: answer): ").strip() or "answer"
                context_col = input("Context column name (default: context): ").strip() or "context"
                return convert_csv_to_legal_format(file_path, output_file, question_col, answer_col, context_col)
            elif file_path.endswith('.jsonl'):
                return convert_jsonl_to_legal_format(file_path, output_file)
            elif file_path.endswith('.json'):
                return convert_json_to_legal_format(file_path, output_file)
            else:
                print("Unsupported file format. Supported: .json, .jsonl, .csv")
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Setup data for legal consultation fine-tuning")
    parser.add_argument("--interactive", action="store_true", help="Interactive data setup")
    parser.add_argument("--url", type=str, help="Direct URL to download data")
    parser.add_argument("--huggingface", type=str, help="HuggingFace dataset name")
    parser.add_argument("--subset", type=str, help="HuggingFace dataset subset")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--validate", action="store_true", help="Validate the final dataset")
    
    args = parser.parse_args()
    
    result_file = None
    
    if args.interactive:
        result_file = setup_legal_data()
    elif args.url:
        result_file = setup_data_from_url(args.url, args.output_dir, "legal_training_data.json")
    elif args.huggingface:
        result_file = process_huggingface_dataset(args.huggingface, 
                                                os.path.join(args.output_dir, "legal_training_data.json"),
                                                args.subset)
    else:
        parser.print_help()
        return
    
    if result_file:
        print(f"\n✅ Data setup completed!")
        print(f"📁 Data file: {result_file}")
        
        # Validate if requested
        if args.validate:
            print("\n🔍 Validating dataset...")
            validate_dataset(result_file)
        
        # Show training command
        print(f"\n🚀 Ready to train! Use this command:")
        print(f"python main.py --data_path {result_file} --create_sample_data")
        
        # Show data statistics
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"\n📊 Dataset Statistics:")
            print(f"   Total examples: {len(data)}")
            if len(data) > 0:
                avg_question_len = sum(len(item.get('question', '')) for item in data) / len(data)
                avg_answer_len = sum(len(item.get('answer', '')) for item in data) / len(data)
                print(f"   Average question length: {avg_question_len:.1f} characters")
                print(f"   Average answer length: {avg_answer_len:.1f} characters")
                
                # Show sample
                print(f"\n📝 Sample entry:")
                sample = data[0]
                print(f"   Question: {sample.get('question', '')[:100]}...")
                print(f"   Answer: {sample.get('answer', '')[:100]}...")
        except Exception as e:
            print(f"Could not read dataset statistics: {e}")
    else:
        print("❌ Data setup failed!")

if __name__ == "__main__":
    main()
