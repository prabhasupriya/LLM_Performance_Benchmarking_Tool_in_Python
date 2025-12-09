import yaml
import pandas as pd
import argparse
from tqdm import tqdm

from modules.loader import load_prompts
from modules.model_runner import load_model, generate_response
from modules.metrics import measure_latency, calculate_throughput, vocab_diversity
from modules.monitor import get_ram_mb, get_gpu_memory
from modules.visualizer import create_latency_plot, create_memory_plot

# ---------------- CLI -----------------
parser = argparse.ArgumentParser(description="LLM Benchmark Tool")
parser.add_argument("--config", required=True)
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

device = cfg["device"]
max_tokens = cfg["max_new_tokens"]


# ---------------- Load dataset -----------------
prompts = load_prompts(cfg["dataset"])

results = []


# ---------------- Benchmark -----------------
for model_cfg in cfg["models"]:

    name = model_cfg["name"]
    model_id = model_cfg["id"]

    print(f"\nLoading {name}")

    tokenizer, model = load_model(model_id, device)

    for prompt in tqdm(prompts):

        ram_before = get_ram_mb()
        gpu_before = get_gpu_memory()

        text, latency = measure_latency(
            generate_response,
            prompt,
            tokenizer,
            model,
            max_tokens
        )

        ram_after = get_ram_mb()
        gpu_after = get_gpu_memory()

        tokens = tokenizer.encode(text)

        throughput = calculate_throughput(len(tokens), latency)
        diversity = vocab_diversity(tokens)

        results.append({
            "model": name,
            "latency_s": latency,
            "throughput_tps": throughput,
            "ram_mb": ram_after - ram_before,
            "gpu_mb": gpu_after,
            "output_length": len(text),
            "vocab_diversity": diversity
        })


# ---------------- Save Results -----------------
df = pd.DataFrame(results)
summary = df.groupby("model").mean().reset_index()

summary.to_csv("reports/results.csv", index=False)

# ---------------- Create charts -----------------
create_latency_plot(summary)
create_memory_plot(summary)

print("\n Benchmark Complete")
print(" Results saved at: reports/results.csv")
print(" Charts saved in reports/")
