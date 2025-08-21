import pyaudio
import numpy as np
import numpy.typing as npt

class AudioData:
    def __init__(
        self,
        frames: bytes,
        format: int = pyaudio.paInt16,
        channels: int = 1,
        rate: int = 16000,
        timestamp: float | None = None
    ):
        assert isinstance(frames, bytes) or isinstance(frames, np.ndarray), f"frames must be of type bytes or numpy ndarray, not {type(frames)}"
        # Format describes how many audio bytes are in each sample
        self.format: int = format
        self.channels: int = channels
        self.rate: int = rate
        self.frames: bytes = bytes()
        if isinstance(frames, np.ndarray):
            self.frames = frames.tobytes()
        else:
            assert isinstance(frames, bytes), f"frames must be of type bytes or numpy ndarray, not {type(frames)}"
            self.frames = frames
        
        self.timestamp: float = timestamp or 0

    def as_bytes(self) -> bytes:
        """ Return raw audio bytes """
        return self.frames
    
    def as_array(self, dtype: npt.DTypeLike | None = None) -> np.ndarray:
        """ Convert bytes to numpy array """
        sample_dtype, _ = self.get_type_info()

        np_array = np.frombuffer(self.frames, dtype=sample_dtype)

        if self.channels > 1:
            np_array = np_array.reshape(-1, self.channels)

        if dtype is not None and dtype != sample_dtype:
            np_array = np_array.astype(dtype)

        return np_array
    
    def __len__(self):
        return self.frame_count()

    def frame_count(self) -> int:
        byte_length: int = len(self.frames)
        sample_size: int = pyaudio.get_sample_size(self.format)

        return byte_length // sample_size // self.channels
    
    def duration(self) -> float:
        return self.frame_count() / self.rate

    def end_time(self) -> float:
        return self.timestamp + self.duration()
    
    def get_type_info(self):
        sample_size = pyaudio.get_sample_size(self.format)
        if sample_size == 1:
            return np.uint8, np.iinfo(np.uint8)
        elif sample_size == 2:
            return np.int16, np.iinfo(np.int16)
        elif sample_size == 4:
            return np.int32, np.iinfo(np.int32)
        elif sample_size == 8:
            return np.float64, np.finfo(np.float64)
        else:
            raise ValueError(f"Unsupported sample size {sample_size}")

    @classmethod
    def sample_width_to_format(cls, sample_width: int):
        return pyaudio.get_format_from_width(sample_width)