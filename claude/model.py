#!/usr/bin/env python3
"""
SaulLM-based Question Answering Model for Legal Consultation
Uses quantized SaulLM with LoRA adapters for efficient fine-tuning
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoConfig,
    GenerationConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType, 
    prepare_model_for_kbit_training,
    PeftModel
)
import logging
from typing import Optional, Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SaulLMQuestionAnswering:
    """
    SaulLM-based Question Answering model for legal consultation.
    Uses 4-bit quantization and LoRA adapters for efficient training.
    """
    
    def __init__(self, 
                 model_name: str = "mradermacher/SaulLM-54B-Instruct-i1-GGUF",
                 use_quantization: bool = False,
                 use_peft: bool = True,
                 device_map: str = "auto",
                 max_memory: Optional[Dict] = None,
                 gguf_file: str = "../SaulLM-54B-Instruct.i1-IQ1_S.gguf"):
        """
        Initialize legal QA model with quantization and LoRA configuration.
        
        Args:
            model_name: HuggingFace model name (default: SaulLM quantized GGUF for testing)
                       Options:
                       - "mradermacher/SaulLM-54B-Instruct-i1-GGUF" (Default, quantized GGUF)
                       - "pierreguillou/gpt2-small-portuguese" (124M params, backup option)
                       - "microsoft/DialoGPT-medium" (345M params, stable fallback)
                       - "Equall/Saul-7B-Instruct-v1" (7B params, production legal model)
            use_quantization: Whether to use 4-bit quantization (recommended for large models)
            use_peft: Whether to use LoRA adapters
            device_map: Device mapping strategy
            max_memory: Maximum memory allocation per device
            gguf_file: GGUF filename for quantized models
        """
        self.model_name = model_name
        self.gguf_file = gguf_file if "GGUF" in model_name else None
        self.use_quantization = use_quantization
        self.use_peft = use_peft
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Auto-adjust settings based on model size
        self._adjust_settings_for_model()
        
        logger.info(f"Initializing Legal QA model: {model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Quantization: {self.use_quantization}")
        logger.info(f"PEFT: {use_peft}")
        
        # Load tokenizer
        self._load_tokenizer()
        
        # Configure quantization
        self.bnb_config = self._configure_quantization() if self.use_quantization else None
        
        # Load model
        self._load_model(device_map, max_memory)
        
        # Configure LoRA
        if use_peft:
            self._configure_peft()
        
        # Set generation config
        self._configure_generation()
    
    def _adjust_settings_for_model(self):
        """Auto-adjust settings based on model name and capabilities"""
        # Model size categories
        small_models = ["gpt2", "distilgpt2", "pierreguillou/gpt2-small-portuguese"]
        medium_models = ["microsoft/DialoGPT-medium", "gpt2-medium"]
        large_models = ["Saul", "7B", "13B", "30B"]
        
        model_lower = self.model_name.lower()
        
        # Determine model size category
        if any(name in model_lower for name in small_models):
            logger.info("Detected small model - optimizing for speed")
            # Small models don't need quantization on most hardware
            if not torch.cuda.is_available():
                self.use_quantization = False
        elif any(name in model_lower for name in large_models):
            logger.info("Detected large model - enabling quantization for memory efficiency")
            # Large models benefit from quantization
            if torch.cuda.is_available():
                self.use_quantization = True
        else:
            logger.info("Detected medium model - using specified settings")
    
    def _load_tokenizer(self):
        """Load and configure tokenizer"""
        logger.info("Loading tokenizer...")
        
        try:
            # Handle GGUF files
            if self.gguf_file:
                logger.info(f"Loading GGUF tokenizer from {self.model_name} with file {self.gguf_file}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    gguf_file=self.gguf_file,
                    trust_remote_code=True,
                    padding_side="right"
                )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    padding_side="right"
                )
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for {self.model_name}: {e}")
            logger.info("Falling back to GPT-2 tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "gpt2",
                padding_side="right"
            )
        
        # Add special tokens for legal QA
        special_tokens = {
            "additional_special_tokens": [
                "[LEGAL_Q]", "[LEGAL_A]", "[CONTEXT]", "[PORTUGAL]", 
                "[IMMIGRATION]", "[VISA]", "[RESIDENCY]", "[CITIZENSHIP]"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        logger.info(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
    
    def _configure_quantization(self) -> BitsAndBytesConfig:
        """Configure 4-bit quantization for memory efficiency"""
        logger.info("Configuring 4-bit quantization...")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    
    def _load_model(self, device_map: str, max_memory: Optional[Dict]):
        """Load the language model with fallback options"""
        logger.info(f"Loading model: {self.model_name}")
        
        # Model loading order (from preferred to fallback)
        models_to_try = [
            self.model_name,  # User specified model
            "pierreguillou/gpt2-small-portuguese",  # Small Portuguese model
            "microsoft/DialoGPT-medium",  # Medium English model
            "gpt2"  # Smallest fallback
        ]
        
        # Remove duplicates while preserving order
        models_to_try = list(dict.fromkeys(models_to_try))
        
        for model_name in models_to_try:
            try:
                logger.info(f"Attempting to load: {model_name}")
                
                # Configure model loading arguments
                model_kwargs = {
                    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                    "low_cpu_mem_usage": True,
                }
                
                # Only add device_map for CUDA
                if torch.cuda.is_available():
                    model_kwargs["device_map"] = device_map
                
                # Add quantization config if enabled and on CUDA
                if self.bnb_config and torch.cuda.is_available():
                    model_kwargs["quantization_config"] = self.bnb_config
                
                # Add memory configuration if provided
                if max_memory:
                    model_kwargs["max_memory"] = max_memory
                
                # Try to load with trust_remote_code for custom models
                if "saul" in model_name.lower() or "custom" in model_name.lower():
                    model_kwargs["trust_remote_code"] = True
                
                # Handle GGUF files
                if model_name == self.model_name and self.gguf_file:
                    logger.info(f"Loading GGUF model from {model_name} with file {self.gguf_file}")
                    model_kwargs["gguf_file"] = self.gguf_file
                    # GGUF models typically come pre-quantized, so disable additional quantization
                    if "quantization_config" in model_kwargs:
                        del model_kwargs["quantization_config"]
                        logger.info("Disabled additional quantization for GGUF model")
                
                # Load the model
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs
                )
                
                # Update model name to successful one
                if model_name != self.model_name:
                    logger.info(f"Successfully loaded fallback model: {model_name}")
                    self.model_name = model_name
                else:
                    logger.info(f"Successfully loaded requested model: {model_name}")
                
                break
                
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                if model_name == models_to_try[-1]:  # Last model in list
                    logger.error("All model loading attempts failed!")
                    raise e
                continue
        
        # Resize token embeddings for new special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Prepare model for training (required for both quantized and non-quantized PEFT)
        if self.use_peft:
            if self.use_quantization and torch.cuda.is_available():
                self.model = prepare_model_for_kbit_training(self.model)
            else:
                # Enable gradient checkpointing for memory efficiency
                self.model.gradient_checkpointing_enable()
                # Ensure model is in training mode
                self.model.train()
    
    def _configure_peft(self):
        """Configure LoRA adapters"""
        logger.info("Configuring LoRA adapters...")
        
        # Auto-detect target modules based on model architecture
        target_modules = self._find_target_modules()
        
        if not target_modules:
            logger.warning("No suitable target modules found, using default linear layers")
            target_modules = ["linear"]  # Fallback
        
        logger.info(f"Using target modules: {target_modules}")
        
        # LoRA configuration for causal language modeling
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # Rank
            lora_alpha=32,  # LoRA scaling parameter
            lora_dropout=0.1,
            bias="none",
            target_modules=target_modules,
        )
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, peft_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        logger.info("LoRA adapters configured")
    
    def _find_target_modules(self):
        """Find suitable target modules for LoRA based on model architecture"""
        target_modules = set()
        
        # Common patterns for different model architectures
        target_patterns = {
            # LLaMA-style models
            "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            # GPT-style models (DialoGPT, GPT-2, etc.)
            "gpt": ["c_attn", "c_proj", "c_fc"],
            # BERT-style models
            "bert": ["query", "key", "value", "dense"],
            # Generic attention patterns
            "generic": ["attn", "attention", "linear", "fc", "proj"]
        }
        
        # Get all module names
        module_names = [name for name, _ in self.model.named_modules()]
        
        # Try to detect model type and find matching modules
        for model_type, patterns in target_patterns.items():
            matches = []
            for pattern in patterns:
                for module_name in module_names:
                    if pattern in module_name.lower():
                        # Get the module class name
                        module_path = module_name.split('.')
                        if len(module_path) > 0:
                            matches.append(module_path[-1])
            
            if matches:
                target_modules.update(matches)
                logger.info(f"Detected {model_type}-style architecture")
                break
        
        # Remove duplicates and filter common ones
        filtered_modules = []
        for module in target_modules:
            if any(keyword in module.lower() for keyword in ['attn', 'proj', 'linear', 'dense', 'fc']):
                filtered_modules.append(module)
        
        return filtered_modules if filtered_modules else list(target_modules)[:4]  # Limit to 4 modules max
    
    def _configure_generation(self):
        """Configure generation parameters"""
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_new_tokens=512,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id if hasattr(self.tokenizer, 'bos_token_id') else None,
        )
    
    def format_qa_prompt(self, question: str, context: str = "") -> str:
        """
        Format question and context into a proper prompt for legal QA.
        
        Args:
            question: Legal question to answer
            context: Additional context for the question
            
        Returns:
            Formatted prompt string
        """
        if context:
            prompt = f"""[LEGAL_Q] Pergunta sobre imigração em Portugal: {question}
[CONTEXT] Contexto: {context}
[LEGAL_A] Resposta detalhada:"""
        else:
            prompt = f"""[LEGAL_Q] Pergunta sobre imigração em Portugal: {question}
[LEGAL_A] Resposta detalhada:"""
        
        return prompt
    
    def generate_answer(self, 
                       question: str, 
                       context: str = "",
                       max_new_tokens: int = 512,
                       temperature: float = 0.7,
                       top_p: float = 0.9) -> str:
        """
        Generate answer for a legal question.
        
        Args:
            question: Legal question to answer
            context: Additional context
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated answer
        """
        try:
            # Format the prompt
            prompt = self.format_qa_prompt(question, context)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True
            )
            
            # Move to device
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    num_return_sequences=1
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the answer part (remove the prompt)
            answer = response[len(prompt):].strip()
            
            # Clean up the answer
            if "[LEGAL_A]" in answer:
                answer = answer.split("[LEGAL_A]")[-1].strip()
            
            return answer if answer else "Desculpe, não consegui gerar uma resposta adequada."
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Erro ao gerar resposta: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        info = {
            "model_name": self.model_name,
            "device": str(self.device),
            "quantization": self.use_quantization,
            "peft": self.use_peft,
            "vocab_size": len(self.tokenizer),
            "model_size": self.model.num_parameters() if hasattr(self.model, 'num_parameters') else "Unknown"
        }
        
        if self.use_peft and hasattr(self.model, 'get_nb_trainable_parameters'):
            trainable, total = self.model.get_nb_trainable_parameters()
            info["trainable_parameters"] = trainable
            info["total_parameters"] = total
            info["trainable_percentage"] = f"{trainable/total*100:.2f}%"
        
        return info
    
    def get_model(self):
        """Get the underlying model"""
        return self.model
    
    def get_tokenizer(self):
        """Get the tokenizer"""
        return self.tokenizer
    
    def save_model(self, output_dir: str):
        """Save the model and tokenizer"""
        logger.info(f"Saving model to {output_dir}")
        
        if self.use_peft:
            # Save only LoRA adapters
            self.model.save_pretrained(output_dir)
        else:
            # Save full model
            self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        logger.info("Model saved successfully")
    
    @classmethod
    def load_trained_model(cls, 
                          model_path: str, 
                          base_model_name: str = "Equall/Saul-7B-Instruct-v1"):
        """
        Load a trained model with LoRA adapters.
        
        Args:
            model_path: Path to the saved LoRA adapters
            base_model_name: Base model name
            
        Returns:
            Loaded model instance
        """
        logger.info(f"Loading trained model from {model_path}")
        
        # Initialize base model
        instance = cls(model_name=base_model_name, use_peft=False)
        
        # Load LoRA adapters
        instance.model = PeftModel.from_pretrained(instance.model, model_path)
        
        logger.info("Trained model loaded successfully")
        return instance
