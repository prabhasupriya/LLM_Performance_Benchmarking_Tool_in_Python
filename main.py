import argparse
import yaml
import os
import pandas as pd
from tqdm import tqdm

from modules.loader import load_model
from modules.data_loader import load_prompts
from modules.metrics import measure_latency, calculate_throughput, vocab_diversity
from modules.monitor import get_ram_usage, get_gpu_usage
from modules.visualizer import create_latency_plot, create_memory_plot


def main(config_path):
    # Ensure reports folder exists
    os.makedirs("reports", exist_ok=True)

    # Load configuration
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load prompts from dataset
    prompts = load_prompts(config["dataset_path"])
    max_tokens = config["generation"]["max_new_tokens"]

    results = []

    # Loop over models
    for model_name in config["models"]:
        print(f"\nLoading model: {model_name}")
        model, tokenizer = load_model(model_name)

        # Loop over prompts
        for prompt in tqdm(prompts, desc=model_name):
            ram_before = get_ram_usage()
            gpu_before = get_gpu_usage()

            def infer():
                inputs = tokenizer(prompt, return_tensors="pt")
                return model.generate(**inputs, max_new_tokens=max_tokens)

            output, latency = measure_latency(infer)

            ram_after = get_ram_usage()
            gpu_after = get_gpu_usage()

            token_count = output.shape[-1]
            throughput = calculate_throughput(token_count, latency)
            diversity = vocab_diversity(
                tokenizer.decode(output[0], skip_special_tokens=True).split()
            )

            results.append({
                "model": model_name,
                "latency_s": latency,
                "throughput_tps": throughput,
                "ram_mb": ram_after - ram_before,
                "gpu_mb": gpu_after - gpu_before,
                "vocab_diversity": diversity
            })

    # Save raw metrics
    df = pd.DataFrame(results)
    df.to_csv("reports/metrics.csv", index=False)

    # Aggregate results
    summary = df.groupby("model").mean().reset_index()

    # Generate plots
    create_latency_plot(summary)
    create_memory_plot(summary)

    # Print summary
    print("\n===== BENCHMARK SUMMARY =====")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Performance Benchmarking Tool")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    main(args.config)
