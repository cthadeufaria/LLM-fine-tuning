#!/usr/bin/env python3
"""
Simple test script for the legal model
"""

import torch
from transformers import BertTokenizer, BertForMaskedLM
from peft import PeftModel
import sys

def test_base_model():
    """Test basic BERT functionality"""
    print("🧪 Testing base BERT model...")
    
    tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    model = BertForMaskedLM.from_pretrained("neuralmind/bert-base-portuguese-cased")
    
    # Test basic tokenization
    text = "Portugal é um país [MASK] para trabalhar."
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
        
    # Get the predicted token for [MASK]
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    predicted_token_id = predictions[0, mask_token_index].argmax(axis=-1)
    predicted_token = tokenizer.decode(predicted_token_id)
    
    print(f"Input: {text}")
    print(f"Predicted token for [MASK]: {predicted_token}")
    print("✅ Base BERT model works correctly!")
    return True

def test_training_data():
    """Check if training data was loaded properly"""
    print("\n📊 Checking training data...")
    
    try:
        import json
        with open("legal_sample_training_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"✅ Training data loaded: {len(data)} samples")
        if len(data) > 0:
            sample = data[0]
            print(f"Sample question: {sample.get('question', 'N/A')[:100]}...")
            print(f"Sample answer: {sample.get('answer', 'N/A')[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return False

def test_peft_model():
    """Test PEFT model loading"""
    print("\n🔧 Testing PEFT model...")
    
    try:
        # Load base model
        from transformers import BertModel
        base_model = BertModel.from_pretrained("neuralmind/bert-base-portuguese-cased")
        
        # Try to load PEFT
        peft_model = PeftModel.from_pretrained(base_model, "./legal_fine_tuned_model")
        print("✅ PEFT model loaded successfully!")
        print(f"Model type: {type(peft_model)}")
        return True
    except Exception as e:
        print(f"❌ Error loading PEFT model: {e}")
        return False

def interactive_test():
    """Simple interactive testing"""
    print("\n💬 Interactive Test Mode")
    print("Enter Portuguese text to test basic tokenization (type 'quit' to exit):")
    
    tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    
    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in ['quit', 'exit', 'sair']:
            break
            
        try:
            tokens = tokenizer.tokenize(user_input)
            token_ids = tokenizer.encode(user_input)
            
            print(f"Tokens: {tokens}")
            print(f"Token IDs: {token_ids}")
            print(f"Decoded: {tokenizer.decode(token_ids)}")
        except Exception as e:
            print(f"Error: {e}")

def main():
    print("🏛️ Legal BERT Model Diagnostic Test")
    print("=" * 50)
    
    # Run all tests
    tests = [
        ("Base BERT Model", test_base_model),
        ("Training Data", test_training_data),
        ("PEFT Model", test_peft_model)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    # Interactive mode if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_test()

if __name__ == "__main__":
    main()
