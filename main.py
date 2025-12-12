import argparse
import yaml
from modules.loader import load_prompts
from modules.model_runner import benchmark_models
from modules.visualizer import create_latency_plot, create_memory_plot
import pandas as pd
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config.yaml")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load prompts
    prompts = load_prompts(config["dataset_path"])

    # Run benchmark
    results_df = benchmark_models(config["models"], prompts, config["generation"]["max_new_tokens"])

    # Save CSV
    os.makedirs("reports", exist_ok=True)
    results_csv_path = "reports/results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"Saved results to {results_csv_path}")

    # Create charts
    create_latency_plot(results_df)
    create_memory_plot(results_df)

    print("\nBenchmarking Completed Successfully!")

if __name__ == "__main__":
    main()
