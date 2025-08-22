import pyaudio
import numpy as np
import numpy.typing as npt
import torch
import torchaudio

from .utils import tensor_to_int16, numpy_to_tensor

class AudioData:
    def __init__(
        self,
        data: bytes,
        format: int = pyaudio.paInt16,
        channels: int = 1,
        rate: int = 16000,
        timestamp: float | None = None
    ):
        assert isinstance(data, bytes) or isinstance(data, np.ndarray), f"frames must be of type bytes or numpy ndarray, not {type(data)}"
        # Format describes how many audio bytes are in each sample
        self.format: int = format
        self.channels: int = channels
        self.rate: int = rate
        self._raw_data: bytes = bytes()
        if isinstance(data, np.ndarray):
            self._raw_data = data.tobytes()
        else:
            assert isinstance(data, bytes), f"frames must be of type bytes or numpy ndarray, not {type(data)}"
            self._raw_data = data
        
        self.timestamp: float = timestamp or 0

    def as_bytes(self) -> bytes:
        """ Return raw audio bytes """
        return self._raw_data
    
    def as_array(self, dtype: npt.DTypeLike | None = None) -> np.ndarray:
        """ Convert bytes to numpy array """
        sample_dtype, _ = self.get_type_info()

        np_array = np.frombuffer(self._raw_data, dtype=sample_dtype)

        if self.channels > 1:
            np_array = np_array.reshape(-1, self.channels)

        if dtype is not None and dtype != sample_dtype:
            np_array = np_array.astype(dtype)

        return np_array

    def as_frames(self) -> list[bytes]:
        """ Convert bytes to list of frames """
        frame_size = pyaudio.get_sample_size(self.format) * self.channels
        return [self._raw_data[i:i + frame_size] for i in range(0, len(self._raw_data), frame_size)]
    
    def __len__(self):
        return self.frame_count()

    def frame_count(self) -> int:
        byte_length: int = len(self._raw_data)
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

    def resample(self, new_rate: int):
        sample_dtype, _ = self.get_type_info()

        # bytes -> numpy array of original sample dtype
        np_data: np.ndarray = np.frombuffer(self._raw_data, dtype=sample_dtype)

        # shape to (time, channels) for multichannel, else (time,)
        if self.channels > 1:
            np_data = np_data.reshape(-1, self.channels)

        # numpy -> torch float32 normalized to [-1, 1]
        audio_tensor: torch.Tensor = numpy_to_tensor(np_data, sample_dtype)

        # torchaudio expects (channels, time)
        if self.channels > 1:
            audio_tensor = audio_tensor.transpose(0, 1)  # (time, ch) -> (ch, time)
        else:
            audio_tensor = audio_tensor.unsqueeze(0)  # (time,) -> (1, time)

        # resample (channels, time)
        print(f"Resampling from {self.rate} to {new_rate}")
        resampler = torchaudio.transforms.Resample(self.rate, new_rate)
        resampled_audio_tensor: torch.Tensor = resampler(audio_tensor)

        # back to int16 numpy in same (channels, time) layout
        data_int16: np.ndarray = tensor_to_int16(resampled_audio_tensor)

        # restore interleaved bytes: (time, channels) -> flattened
        if self.channels > 1:
            data_int16 = np.transpose(data_int16, (1, 0))  # (time, channels)
            data_bytes = data_int16.reshape(-1).tobytes()
        else:
            data_bytes = data_int16.squeeze(0).tobytes()
        
        # convert back to audio data
        audio_data = AudioData(data_bytes, format=self.format, channels=self.channels, rate=new_rate)
        return audio_data
        
    @classmethod
    def sample_width_to_format(cls, sample_width: int):
        return pyaudio.get_format_from_width(sample_width)