from transformers import Trainer, TrainingArguments
from dataset import PLSDataset
import logging
import transformers
import datasets
import sys


class UnsupervisedTrainer(Trainer):
    def __init__(self, wraper):
        training_args = TrainingArguments(
            output_dir="./finetuned-llama",
            per_device_train_batch_size=1,        # Reduced from 2
            gradient_accumulation_steps=4,        # Reduced from 16
            learning_rate=2e-5,
            num_train_epochs=1,                   # Reduced from 3
            save_steps=50,                        # Reduced from 500
            logging_steps=1,                      # Log every step for debugging
            fp16=False,   # since CPU
            dataloader_pin_memory=False,          # Disable pin_memory for CPU
            max_steps=10,                         # Even smaller for debugging
            logging_first_step=True,              # Log the first step
            report_to=None,                       # Disable wandb for now
        )

        logger = logging.getLogger(__name__)

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        log_level = training_args.get_process_log_level()
        logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)

        self.dataset = PLSDataset(wraper.tokenizer)

        self.trainer = Trainer(
            model=wraper.model,
            args=training_args,
            train_dataset=self.dataset,
            tokenizer=wraper.tokenizer,
        )

    def train_model(self):
        print("🔍 Starting training debug...")
        print(f"📊 Dataset size: {len(self.dataset)}")
        print(f"🤖 Model type: {type(self.trainer.model)}")
        print(f"🔧 Training args: batch_size={self.trainer.args.per_device_train_batch_size}, max_steps={self.trainer.args.max_steps}")
        
        # Test dataset access
        print("🧪 Testing dataset access...")
        try:
            sample = self.dataset[0]
            print(f"✅ Sample keys: {sample.keys()}")
            print(f"✅ Input shape: {sample['input_ids'].shape}")
        except Exception as e:
            print(f"❌ Dataset error: {e}")
            return
        
        # Test model forward pass
        print("🧪 Testing model forward pass...")
        try:
            import torch
            with torch.no_grad():
                sample_batch = {k: v.unsqueeze(0) for k, v in sample.items()}
                output = self.trainer.model(**sample_batch)
                print(f"✅ Model forward pass successful, loss: {output.loss}")
        except Exception as e:
            print(f"❌ Model forward error: {e}")
            return
        
        print("🚀 Starting actual training...")
        self.trainer.train()
