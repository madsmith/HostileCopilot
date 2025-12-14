#!/usr/bin/env python
"""Install PyTorch with appropriate CUDA version based on system capabilities."""
import subprocess
import sys
import re
import platform


def get_cuda_version() -> str | None:
    """Get max CUDA version from nvidia-smi."""
    if platform.system() == "Darwin":
        return None  # macOS uses MPS, not CUDA
    
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True
        )
        # Parse "CUDA Version: 12.4" from output
        match = re.search(r"CUDA Version:\s*(\d+\.\d+)", result.stdout)
        if match:
            return match.group(1)
    except FileNotFoundError:
        pass
    
    return None


def get_pytorch_index(cuda_version: str | None) -> str:
    """Map CUDA version to PyTorch wheel index."""
    if cuda_version is None:
        return "https://download.pytorch.org/whl/cpu"
    
    major, minor = map(int, cuda_version.split("."))
    version = major * 10 + minor  # 12.4 -> 124
    
    # Available indexes: cu118, cu121, cu124, cu126, cu128, cu130
    if version >= 130:
        return "https://download.pytorch.org/whl/cu130"
    if version >= 128:
        return "https://download.pytorch.org/whl/cu128"
    elif version >= 126:
        return "https://download.pytorch.org/whl/cu126"
    elif version >= 124:
        return "https://download.pytorch.org/whl/cu124"
    elif version >= 121:
        return "https://download.pytorch.org/whl/cu121"
    elif version >= 118:
        return "https://download.pytorch.org/whl/cu118"
    else:
        return "https://download.pytorch.org/whl/cpu"


def main():
    cuda_version = get_cuda_version()
    index_url = get_pytorch_index(cuda_version)
    
    if cuda_version:
        print(f"Detected CUDA {cuda_version}")
    else:
        print("No CUDA detected, installing CPU version")
    
    print(f"Using index: {index_url}")
    
    try:
        print(sys.executable)
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", 
            "torch", "torchvision", "torchaudio",
            "--index-url", index_url
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to install PyTorch: {e}")
        sys.exit(1)
    
    print("Done!")


if __name__ == "__main__":
    main()