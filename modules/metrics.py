import time

def measure_latency(func, *args):

    start = time.perf_counter()
    result = func(*args)
    end = time.perf_counter()

    latency = end-start

    return result, latency


def calculate_throughput(tokens, latency):
    return tokens / latency if latency > 0 else 0


def vocab_diversity(tokens):
    return len(set(tokens))/max(len(tokens),1)
