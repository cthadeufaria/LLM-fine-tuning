# 🏛️ Legal BERT Data Setup Guide

This guide shows you how to use files downloaded from the internet to fine-tune your legal consultation model.

## 🚀 Quick Start Options

### Option 1: Interactive Data Setup
```bash
python setup_data.py --interactive
```
This will guide you through selecting and setting up your data source.

### Option 2: Use Sample Data (for testing)
```bash
python main.py --create_sample_data --data_path legal_sample_data.json --max_steps 50
```

### Option 3: Download from URL
```bash
# Download and convert automatically
python data_utils.py download "https://example.com/legal_data.json"

# Then train with the converted data
python main.py --data_path ./data/training_data.json
```

### Option 4: Use HuggingFace Dataset
```bash
# Download HuggingFace dataset
python data_utils.py huggingface "dataset_name" "./data/legal_data.json"

# Train with the data
python main.py --data_path ./data/legal_data.json
```

## 📋 Supported Data Formats

### 1. JSON Format
```json
[
  {
    "question": "Como renovar o título de residência?",
    "context": "Título expira em 2 meses",
    "answer": "A renovação deve ser solicitada entre 30 dias antes..."
  }
]
```

Convert: `python data_utils.py convert json input.json output.json`

### 2. JSONL Format
```jsonl
{"question": "...", "answer": "...", "context": "..."}
{"question": "...", "answer": "...", "context": "..."}
```

Convert: `python data_utils.py convert jsonl input.jsonl output.json`

### 3. CSV Format
```csv
question,answer,context
"Como solicitar visto?","Para solicitar visto...","Cidadão brasileiro"
```

Convert: `python data_utils.py convert csv input.csv output.json --question_col question --answer_col answer`

### 4. Various Field Mappings
The converter automatically detects these field names:
- **Questions**: `question`, `input`, `instruction`, `query`, `pergunta`
- **Answers**: `answer`, `output`, `response`, `resposta`, `text`
- **Context**: `context`, `background`, `contexto`, `passage`

## 🌐 Real Dataset Examples

### Legal Datasets You Can Use:

1. **Portuguese Legal Q&A**:
   ```bash
   # Example with a real Portuguese legal dataset
   python setup_data.py --url "https://your-legal-dataset.com/pt_legal.json"
   ```

2. **HuggingFace Legal Datasets**:
   ```bash
   # Brazilian legal dataset (example)
   python data_utils.py huggingface "brazilian_legal_qa" legal_data.json
   
   # Portuguese government Q&A
   python data_utils.py huggingface "pt_gov_qa" legal_data.json
   ```

3. **Government Open Data**:
   ```bash
   # SEF/AIMA FAQ data (example URL)
   python data_utils.py download "https://www.sef.pt/api/faq.json"
   ```

4. **Legal Text Collections**:
   ```bash
   # Legal document collections
   python data_utils.py download "https://legal-texts.pt/immigration_qa.zip"
   ```

## 🔧 Advanced Data Processing

### 1. Multiple Data Sources
```bash
# Download and convert multiple sources
python data_utils.py download "https://source1.com/data.json" --output_name data1.json
python data_utils.py download "https://source2.com/data.csv" --output_name data2.json

# Combine datasets
python -c "
import json
with open('./data/data1.json') as f1, open('./data/data2.json') as f2:
    data1, data2 = json.load(f1), json.load(f2)
    combined = data1 + data2
    with open('./data/combined_legal_data.json', 'w') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
print('Combined dataset ready!')
"
```

### 2. Data Validation and Cleaning
```bash
# Validate your dataset
python data_utils.py validate ./data/legal_data.json

# Split into train/validation/test
python data_utils.py split ./data/legal_data.json --train_ratio 0.8 --val_ratio 0.1
```

### 3. Custom Field Mapping for CSV
```bash
# If your CSV has different column names
python data_utils.py convert csv legal_faq.csv legal_data.json \
  --question_col "pergunta" \
  --answer_col "resposta" \
  --context_col "categoria"
```

## 📊 Data Quality Tips

### 1. Check Your Data
```bash
python -c "
import json
with open('./data/legal_data.json') as f:
    data = json.load(f)
    print(f'Total examples: {len(data)}')
    print(f'Average question length: {sum(len(d[\"question\"]) for d in data)/len(data):.1f}')
    print(f'Average answer length: {sum(len(d[\"answer\"]) for d in data)/len(data):.1f}')
    print(f'Examples with context: {sum(1 for d in data if d.get(\"context\"))}')
"
```

### 2. Filter and Clean
```bash
# Remove short answers (less than 50 characters)
python -c "
import json
with open('./data/legal_data.json') as f:
    data = json.load(f)
    filtered = [d for d in data if len(d.get('answer', '')) >= 50]
    print(f'Filtered from {len(data)} to {len(filtered)} examples')
    with open('./data/legal_data_filtered.json', 'w') as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
"
```

## 🎯 Complete Workflow Example

```bash
# 1. Setup your workspace
./setup.sh

# 2. Get your data (choose one):
# Option A: Interactive setup
python setup_data.py --interactive

# Option B: Direct download
python data_utils.py download "https://your-legal-dataset.com/data.json"

# Option C: HuggingFace dataset
python data_utils.py huggingface "legal_dataset_name" "./data/legal_data.json"

# 3. Validate the data
python data_utils.py validate ./data/legal_data.json

# 4. Split for training
python data_utils.py split ./data/legal_data.json

# 5. Start training
python main.py --data_path ./data/legal_data_train.json --num_epochs 5

# 6. Test your model
python inference.py \
  --base_model neuralmind/bert-base-portuguese-cased \
  --peft_model ./legal_fine_tuned_model \
  --interactive
```

## 🔍 Finding Legal Datasets

### Portuguese Immigration Law Sources:
1. **SEF/AIMA Official Data**: Government immigration agency
2. **Portuguese Legal Databases**: Academic and legal institutions
3. **Community Q&A Sites**: Stack Overflow equivalents for legal questions
4. **Legal Firm FAQs**: Immigration lawyer websites
5. **Government Open Data**: Portuguese open data initiatives

### Search Terms:
- "Portuguese immigration law dataset"
- "legal Q&A Portugal"
- "immigration FAQ dataset"
- "Portuguese legal text mining"
- "SEF AIMA FAQ data"

## ⚠️ Important Notes

1. **Data Quality**: Ensure your data is accurate and from reliable sources
2. **Legal Compliance**: Respect copyright and data usage rights
3. **Language**: Make sure data is in Portuguese for best results
4. **Domain Relevance**: Focus on immigration law for specialized performance
5. **Validation**: Always validate legal advice with qualified professionals

## 🆘 Troubleshooting

### Common Issues:
1. **Download Fails**: Check URL and internet connection
2. **Format Not Recognized**: Use manual conversion commands
3. **Encoding Issues**: Ensure UTF-8 encoding for Portuguese text
4. **Memory Issues**: Split large datasets before processing
5. **Field Mapping**: Check column names in CSV files

### Get Help:
```bash
# Show all available commands
python data_utils.py --help
python setup_data.py --help
python main.py --help

# Validate your setup
python -c "from data_utils import validate_dataset; validate_dataset('your_file.json')"
```
