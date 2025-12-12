import psutil

try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    nvmlInit()
    GPU_AVAILABLE = True
    handle = nvmlDeviceGetHandleByIndex(0)
except:
    GPU_AVAILABLE = False


def get_ram_usage():
    return psutil.Process().memory_info().rss / (1024 ** 2)


def get_gpu_usage():
    if not GPU_AVAILABLE:
        return 0
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    return mem_info.used / (1024 ** 2)
