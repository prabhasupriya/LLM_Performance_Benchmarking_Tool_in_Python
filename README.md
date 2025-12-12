# LLM Performance Benchmarking Tool

A CLI-based tool to benchmark open-source Large Language Models (LLMs) on **inference latency, throughput, memory usage, and output quality**.


## ⚡ Tool Features

✅ Config-driven CLI interface  
✅ HuggingFace model evaluation  
✅ CPU + GPU benchmarking  
✅ 8-bit memory optimization support  
✅ Memory & latency profiling  
✅ Visualization charts (latency & memory)  
✅ CSV metrics export  



## 🧠 Models Tested

- DistilGPT2  
- Pythia-2.8B (tested on Colab GPU)  
- Falcon-7B (tested on Colab GPU)  
- Tiny-GPT2, Bloom-560M, GPT-Neo-1.3B (optional/custom)



## 📊 Outputs

| File | Purpose |
|------|---------|
| `results.csv` | Benchmark summary with latency & memory metrics |
| `latency_plot.png` | Latency comparison chart |
| `memory_plot.png` | GPU/CPU memory usage chart |
| `model_responses.json` | Prompt-response pairs from all models |



## 🚀 How to Run

```bash


# Install dependencies
pip install -r requirements.txt

# Run benchmark
python main.py --config config.yaml
