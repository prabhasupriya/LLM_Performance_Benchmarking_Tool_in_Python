import psutil
import pynvml


def get_ram_mb():
    return psutil.Process().memory_info().rss / 1024**2


def get_gpu_memory():

    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        return info.used / 1024**2
    except:
        return 0
