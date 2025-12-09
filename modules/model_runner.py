from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model(model_id, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if device=="cuda" else None,
        torch_dtype=torch.float16 if device=="cuda" else torch.float32,
        load_in_8bit=True if device=="cuda" else False
    )

    model = model.to(device)
    model.eval()

    return tokenizer, model


def generate_response(prompt, tokenizer, model, max_tokens):

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    return text[len(prompt):].strip()
