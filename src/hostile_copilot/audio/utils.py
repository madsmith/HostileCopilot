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