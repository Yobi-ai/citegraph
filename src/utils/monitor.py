from typing import Optional, Tuple

import psutil


def get_cpu_usage() -> float:
    return float(psutil.cpu_percent())


def get_memory_usage() -> Tuple[float, float]:
    mem = psutil.virtual_memory()
    return mem.used / (1024**3), mem.total / (1024**3)


def log_system_metrics(epoch: Optional[int] = None) -> None:
    cpu = get_cpu_usage()
    used_mem, total_mem = get_memory_usage()
    print(f"[Epoch {epoch}] CPU: {cpu}% | RAM: {used_mem:.2f} / {total_mem:.2f} GB")
