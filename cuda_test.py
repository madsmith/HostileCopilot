#!/usr/bin/env python3

import time
import torch

def warmup(device):
    dummy = torch.randn((1024, 1024), device=device)
    _ = dummy @ dummy
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

print("=== CUDA TEST ===")

# 1. Basic availability
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)

if not torch.cuda.is_available():
    print("CUDA NOT AVAILABLE — PyTorch is running CPU-only.")
    raise SystemExit()

# 2. Device info
device_index = torch.cuda.current_device()
print("CUDA Device Index:", device_index)
print("CUDA Device Name:", torch.cuda.get_device_name(device_index))

# 3. Initialize CUDA
torch.cuda.init()
print("CUDA runtime initialized.")

device = torch.device("cuda")

# 4. Actual GPU compute test
warmup(device)

print("\nRunning a 4000x4000 GEMM (matrix multiply) on GPU...")
torch.cuda.synchronize()

x = torch.randn((4000, 4000), device=device)
y = torch.randn((4000, 4000), device=device)

torch.cuda.synchronize()

start = time.time()
z = x @ y  # heavy GPU operation
torch.cuda.synchronize()

elapsed = time.time() - start

print(f"Matrix multiply completed in {(elapsed * 1000):.1f} ms")
if elapsed < 0.05:
    print("Success")
else:
    print("Error: Matrix multiply took too long.")

