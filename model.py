from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch.nn as nn
import torch


class ModelWrapper(nn.Module):
    """
    https://stackoverflow.com/questions/76060541/further-finetune-a-peft-lora-finetuned-causallm-model
    """
    
    def __init__(self, model_id: str):
        super(ModelWrapper, self).__init__()
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,           
            bnb_4bit_quant_type="nf4",    
            bnb_4bit_use_double_quant=True, 
            bnb_4bit_compute_dtype=torch.bfloat16, 
        )

        if model_id == "Equall/SaulLM-54B-Instruct":
            model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)

        elif model_id == "mradermacher/SaulLM-54B-Instruct-i1-GGUF":
            filename = "SaulLM-54B-Instruct.i1-IQ1_S.gguf"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                gguf_file=filename,
                quantization_config=nf4_config
            )

        elif model_id == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="cpu",        # force CPU
                torch_dtype="float16",    # mixed precision
                quantization_config=nf4_config
            )

        lora_config = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        )

        self.model = get_peft_model(model, lora_config)
        self.model.print_trainable_parameters()