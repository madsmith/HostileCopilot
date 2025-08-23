from abc import ABC, abstractmethod
import logging
import numpy as np
from numpy.typing import DTypeLike
import pyaudio

logger = logging.getLogger(__name__)

class AudioBufferBase(ABC):
    def __init__(self, sample_rate: int = 16000, format: int = pyaudio.paInt16, channels: int = 1):
        self._sample_rate: int = sample_rate
        self._audio_format: int = format
        self._channels: int = channels
        self._frame_size: int = pyaudio.get_sample_size(format) * self._channels

    def get_channels(self) -> int:
        return self._channels
    
    def get_sample_rate(self) -> int:
        return self._sample_rate
    
    def get_audio_format(self) -> int:
        return self._audio_format
    
    def get_sample_size(self) -> int:
        return pyaudio.get_sample_size(self._audio_format)

    @abstractmethod
    def append(self, chunk) -> None:
        pass

    @abstractmethod
    def get_bytes(self) -> bytes:
        pass

    @abstractmethod
    def byte_count(self) -> int:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    def __len__(self) -> int:
        return self.frame_count()

    def frame_count(self) -> int:
        return self.byte_count() // self._frame_size

    def get_frames(self) -> np.ndarray:
        np_dtype = self._frame_size_to_numpy()
        audio_data = np.frombuffer(self.get_bytes(), dtype=np_dtype)

        if self._channels > 1:
            audio_data = audio_data.reshape(-1, self._channels)
        return audio_data

    def get_duration_ms(self) -> float:
        return self.frame_count() / self._sample_rate * 1000

    def _frame_size_to_numpy(self) -> DTypeLike:
        # convert frame size to numpy dtype
        if self._frame_size == 1:
            return np.uint8
        elif self._frame_size == 2:
            return np.int16
        elif self._frame_size == 4:
            return np.int32
        else:
            raise ValueError(f"Unsupported frame size {self._frame_size}")

class AudioBuffer(AudioBufferBase):
    def __init__(self, sample_rate: int = 16000, format: int = pyaudio.paInt16, channels: int = 1):
        super().__init__(sample_rate=sample_rate, format=format, channels=channels)
        self.chunks = []

    def append(self, chunk):
        if isinstance(chunk, np.ndarray):
            chunk = chunk.tobytes()

        chunk_size = len(chunk)
        if chunk_size % self._frame_size != 0:
            logger.warning(f"Chunk size {chunk_size} is not a multiple of frame size {self._frame_size}")
            # drop last few samples
            chunk = chunk[:-(chunk_size % self._frame_size)]

        self.chunks.append(chunk)

    def get_bytes(self) -> bytes:
        return b"".join(self.chunks)

    def byte_count(self) -> int:
        return sum(len(chunk) for chunk in self.chunks)

    def clear(self):
        self.chunks.clear()