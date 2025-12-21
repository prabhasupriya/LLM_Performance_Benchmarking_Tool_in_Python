LLM Performance Benchmarking Tool

A CLI-based benchmarking framework designed to evaluate and compare open-source Large Language Models (LLMs) based on inference latency, throughput, memory consumption, and output quality.
The tool helps understand performance–resource trade-offs across models of different sizes.

📌 Project Objectives

To measure inference performance of LLMs in a standardized manner

To compare lightweight vs large models

To analyze latency, throughput, and memory usage

To generate visual and tabular benchmark reports

To support CPU and GPU environments

⚙️ Key Features

✅ Config-driven benchmarking via YAML
✅ HuggingFace Transformer model support
✅ CPU & GPU inference benchmarking
✅ RAM and GPU memory monitoring
✅ Latency & throughput measurement
✅ Vocabulary diversity as output quality metric
✅ Automatic CSV report generation
✅ Visualization plots for performance comparison

🧠 Models Evaluated
Model	Size	Environment
sshleifer/tiny-gpt2	~6M params	Local CPU
bigscience/bloom-560m	560M params	Local CPU

 ⚠️ Optional / Future Models
The following models were considered but not benchmarked in this project due to hardware constraints:
- EleutherAI/gpt-neo-1.3B (requires High-RAM system)
- DistilGPT2 (medium, optional)
- Falcon-7B (large, Colab GPU)
- Pythia-2.8B (large, Colab GPU)



## Project Structure
LLM-Benchmarking-Tool/
│
├── data/
│   ├── prompts.jsonl
│   └── again_quiz.jsonl
│
├── modules/
│   ├── data_loader.py     # Loads prompt datasets
│   ├── loader.py          # Loads HuggingFace models
│   ├── metrics.py         # Latency, throughput, diversity
│   ├── monitor.py         # RAM & GPU usage tracking
│   └── visualizer.py      # Performance plots
│
├── reports/
│   ├── metrics.csv
│   ├── latency_plot.png
│   └── memory_plot.png
│
├── config.yaml
├── main.py
├── requirements.txt
└── README.md

## Output Files
File	Description
metrics.csv	Detailed benchmark metrics per prompt
latency_plot.png	Average inference latency comparison
memory_plot.png	Average RAM/GPU memory usage
Console Summary	Mean metrics per model
## Metrics Explained
🔹 Latency (seconds)

Time taken to generate output for a prompt.

🔹 Throughput (tokens/sec)

Number of tokens generated per second.

🔹 RAM Usage (MB)

Difference in memory before and after inference.

🔹 GPU Memory (MB)

Measured via NVIDIA NVML (if GPU available).

🔹 Vocabulary Diversity

Ratio of unique tokens to total tokens in output.

## Sample Observation

BLOOM-560M consumes significantly more memory than Tiny-GPT2, confirming the expected trade-off between model size and resource usage.
Tiny-GPT2 offers fast inference with minimal memory, while BLOOM delivers richer outputs at higher computational cost.

## Setup Instructions
1️⃣ Create Virtual Environment
python -m venv venv

2️⃣ Activate Environment (Windows)
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

## Running the Benchmark
python main.py --config config.yaml

## Configuration (config.yaml)
models:
  - sshleifer/tiny-gpt2
  - bigscience/bloom-560m

dataset_path: data/prompts.jsonl

generation:
  max_new_tokens: 50

### System Compatibility
Environment	Supported
Windows	YES
Linux	YES
CPU	YES
NVIDIA GPU	YES
Google Colab	YES
### Future Enhancements

Batch inference support

Energy consumption tracking

Per-token latency analysis

Web-based dashboard

BLEU / ROUGE evaluation

### Conclusion

This tool provides a scalable, modular, and reproducible benchmarking pipeline for evaluating LLMs.
It enables informed decisions when choosing models based on performance vs resource constraints, making it valuable for both academic research and practical deployment scenarios.