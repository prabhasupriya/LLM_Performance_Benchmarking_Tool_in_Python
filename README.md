# LLM Performance Benchmarking Tool

This project benchmarks open-source language models using automated metrics to compare:

- Inference latency
- Throughput (tokens per second)
- RAM & GPU memory usage
- Output quality statistics

## Tool Features

✅ Config-driven CLI interface  
✅ HuggingFace model evaluation  
✅ CPU + GPU benchmarking  
✅ Memory profiling  
✅ Visualization charts  
✅ CSV metrics export  


## Models Tested

- DistilGPT2  
- Pythia-2.8B *(Colab GPU run)*  
- Falcon-7B *(Colab GPU run)*

---

## Outputs

| File | Purpose |
|------|---------|
| results.csv | Performance summary |
| latency_plot.png | Latency comparison |
| memory_plot.png | GPU memory usage chart |

---

## How to Run

```bash
python main.py --config config.yaml
