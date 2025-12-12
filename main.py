# main.py
import importlib
import modules.model_runner as mr
import yaml, json, torch, os

# Reload module to avoid cache issues in VSCode
importlib.reload(mr)

# Assign functions
load_model = mr.load_model
generate_response = mr.generate_response

print("Modules reloaded successfully!")

# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

models = cfg["models"]
dataset_path = cfg["dataset_path"]
max_tokens = cfg["generation"]["max_new_tokens"]

# Load prompts from JSONL
prompts = []
with open(dataset_path, "r") as f:
    for line in f:
        data = json.loads(line)
        if "prompt" in data:
            prompts.append(data["prompt"])

print(f"Loaded {len(prompts)} prompts. First 3 prompts: {prompts[:3]}")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Create reports folder if not exists
os.makedirs("reports", exist_ok=True)

# Dictionary to save results
all_responses = {}

# Run models
for model_id in models:
    print(f"\nLoading model: {model_id}")
    tokenizer, model = load_model(model_id, device)
    print(f"Model {model_id} loaded successfully!\n")
    
    model_responses = []
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}: {prompt}")
        response = generate_response(prompt, tokenizer, model, max_tokens)
        print(f"Response:\n{response}\n")
        model_responses.append({
            "prompt": prompt,
            "response": response
        })
    
    all_responses[model_id] = model_responses

# Save all responses
output_path = "reports/model_responses.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_responses, f, indent=4, ensure_ascii=False)

print(f"All responses saved in {output_path}")
