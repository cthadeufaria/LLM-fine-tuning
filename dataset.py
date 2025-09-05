from torch.utils.data import Dataset
from datasets import load_dataset


class PLSDataset(Dataset):
    def __init__(self, tokenizer, dataset_id: str = "rufimelo/PortugueseLegalSentences-v3"):
        self.data = load_dataset(dataset_id)
        self.tokenizer = tokenizer
        # LIMIT DATASET SIZE FOR CPU TESTING
        self.max_samples = 100  # Only use first 100 samples for testing

    def __len__(self):
        return min(len(self.data['train']), self.max_samples)

    def __getitem__(self, idx):
        if idx == 0:  # Debug first item
            print(f"🔍 Loading sample {idx}")
        
        text = self.data['train'][idx]['text']
        
        if idx == 0:
            print(f"📝 Text preview: {text[:100]}...")

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        
        if idx == 0:
            print(f"🔤 Tokenized input_ids shape: {encoding['input_ids'].shape}")
        
        # Return properly formatted dictionary for causal LM
        result = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()  # For causal LM, labels = input_ids
        }
        
        if idx == 0:
            print(f"✅ Final result shapes: input_ids={result['input_ids'].shape}")
        
        return result