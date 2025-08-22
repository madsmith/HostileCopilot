import numpy as np
import torch

def tensor_to_int16(audio_tensor: torch.Tensor) -> np.ndarray:
    # Clamp values to avoid overflows
    audio_clamped = torch.clamp(audio_tensor, -1.0, 1.0)
    # Convert to int16 range

    max_int16 = np.iinfo(np.int16).max
    audio_int16 = (audio_clamped * max_int16).to(torch.int16)
    audio_np: np.ndarray = audio_int16.numpy()
    return audio_np

def numpy_to_tensor(audio_np: np.ndarray, dtype: np.dtype) -> torch.Tensor:
    # If input dtype is integer, scale to [-1, 1] by dtype max.
    # If input dtype is floating, assume already in [-1, 1] and just clamp.
    if np.issubdtype(dtype, np.integer):
        max_size = np.iinfo(dtype).max
        audio_scaled = audio_np.astype(np.float32) / float(max_size)
    elif np.issubdtype(dtype, np.floating):
        audio_scaled = audio_np.astype(np.float32)
    else:
        # Fallback: attempt float32 cast without scaling
        audio_scaled = audio_np.astype(np.float32)

    audio_tensor = torch.tensor(audio_scaled, dtype=torch.float32)
    audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)
    return audio_tensor