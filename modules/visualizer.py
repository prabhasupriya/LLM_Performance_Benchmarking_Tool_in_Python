import matplotlib.pyplot as plt
import pandas as pd
import os


def create_latency_plot(summary_df):
    plt.figure()
    plt.bar(summary_df["model"], summary_df["latency_s"])
    plt.xlabel("Model")
    plt.ylabel("Average Latency (seconds)")
    plt.title("Average Latency per Model")
    plt.xticks(rotation=20)
    plt.tight_layout()

    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/latency_plot.png")
    plt.close()


def create_memory_plot(summary_df):
    plt.figure()
    plt.bar(summary_df["model"], summary_df["ram_mb"])
    plt.xlabel("Model")
    plt.ylabel("Average RAM Usage (MB)")
    plt.title("Average RAM Usage per Model")
    plt.xticks(rotation=20)
    plt.tight_layout()

    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/memory_plot.png")
    plt.close()
