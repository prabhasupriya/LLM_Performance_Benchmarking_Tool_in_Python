# modules/model_runner.py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

def load_model(model_id, device):
    """
    Load a HuggingFace model with optional 8-bit quantization on GPU.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if device == "cuda":
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True  # Handles 8-bit quantization automatically
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quant_config
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=None,
            torch_dtype=torch.float32
        )
        model = model.to(device)

    model.eval()
    return tokenizer, model


def generate_response(prompt, tokenizer, model, max_tokens):
    """
    Generate text using the model with sampling options.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,      # Enable sampling
        temperature=0.7,     # Randomness
        top_k=50,            # Top-k sampling
        top_p=0.9            # Nucleus sampling
    )
    
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text[len(prompt):].strip()
