import matplotlib.pyplot as plt


def create_latency_plot(df):

    plt.figure()
    plt.bar(df["model"], df["latency_s"])
    plt.title("Average Latency (seconds)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("reports/latency_plot.png")


def create_memory_plot(df):

    plt.figure()
    plt.bar(df["model"], df["gpu_mb"])
    plt.title("GPU Memory Usage (MB)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("reports/memory_plot.png")
