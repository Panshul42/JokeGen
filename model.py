from transformers import GPT2LMHeadModel, AutoTokenizer, BitsAndBytesConfig
import torch

def load_model_and_tokenizer(model_name="gpt2-medium"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configure 4-bit loading for smaller memory footprint
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Use 4-bit quantization
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization
    model = GPT2LMHeadModel.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto"  # Automatically handles device placement
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Configure for training
    model.config.use_cache = False  # Disable cache for gradient checkpointing
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer