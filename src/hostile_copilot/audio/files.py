import asyncio
from pathlib import Path
import pyaudio
import wave

from .audio_data import AudioData

def load_wave_file(file_path: str | Path) -> AudioData:
    if isinstance(file_path, Path):
        file_path = str(file_path)

    with wave.open(file_path, 'rb') as wf:
        sample_width = wf.getsampwidth()
        frame_bytes = wf.readframes(wf.getnframes())
        audio_data = AudioData(
            frame_bytes,
            format=AudioData.sample_width_to_format(sample_width),
            channels=wf.getnchannels(),
            rate=wf.getframerate()
        )

    return audio_data

def save_wave_file(audio_data: AudioData, file_path: str | Path) -> None:
    if isinstance(file_path, Path):
        file_path = str(file_path)

    channels = audio_data.channels
    sample_width = audio_data.format
    rate = audio_data.rate
    data = audio_data._raw_data
    
    sample_width = pyaudio.get_sample_size(sample_width)

    with wave.open(file_path, 'wb') as wf:
        print(f"Saving wave file: {file_path}, {channels} channels, {sample_width} bytes per sample, {rate} Hz")
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(data)

async def save_wave_file_async(audio_data: AudioData, file_path: str | Path) -> None:
    await asyncio.to_thread(save_wave_file, audio_data, file_path)