import psutil


def get_cpu_usage():
    return psutil.cpu_percent()

def get_memory_usage():
    mem = psutil.virtual_memory()
    return mem.used / (1024 ** 3), mem.total / (1024 ** 3)

def log_system_metrics(epoch=None):
    cpu = get_cpu_usage()
    used_mem, total_mem = get_memory_usage()
    print(f"[Epoch {epoch}] CPU: {cpu}% | RAM: {used_mem:.2f} / {total_mem:.2f} GB")
