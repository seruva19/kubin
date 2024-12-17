import torch


def gpu_memory_usage(device):
    memory = torch.cuda.memory_allocated(device) / 1024**3
    max_memory = torch.cuda.max_memory_allocated(device) / 1024**3
    max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    print(
        f"memory: {memory=:.3f}, max: {max_memory=:.3f}, reserved {max_reserved=:.3f}"
    )
