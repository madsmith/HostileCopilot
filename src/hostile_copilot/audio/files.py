import asyncio
from pathlib import Path
import pyaudio
from pydub import AudioSegment
import wave

from .audio_data import AudioData
from .buffers import AudioBuffer

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

def save_wave_file(audio_data: AudioData | AudioBuffer | AudioSegment, file_path: str | Path) -> None:
    if isinstance(file_path, Path):
        file_path = str(file_path)

    channels = audio_data.channels
    sample_size = audio_data.format
    rate = audio_data.rate
    data = audio_data._raw_data

    if isinstance(audio_data, AudioData):
        channels = audio_data.channels
        sample_size, _ = audio_data.get_type_info()
        rate = audio_data.rate
        data = audio_data._raw_data

    elif isinstance(audio_data, AudioBuffer):
        channels = audio_data.get_channels()
        sample_size = audio_data.get_sample_size()
        rate = audio_data.get_sample_rate()
        data = audio_data.get_bytes()

    elif isinstance(audio_data, AudioSegment):
        channels = audio_data.channels
        sample_size = audio_data.sample_width
        rate = audio_data.frame_rate
        data = audio_data.raw_data

    with wave.open(file_path, 'wb') as wf:
        print(f"Saving wave file: {file_path}, {channels} channels, {sample_size} bytes per sample, {rate} Hz")
        wf.setnchannels(channels)
        wf.setsampwidth(sample_size)
        wf.setframerate(rate)
        wf.writeframes(data)

async def save_wave_file_async(audio_data: AudioData | AudioBuffer | AudioSegment, file_path: str | Path) -> None:
    await asyncio.to_thread(save_wave_file, audio_data, file_path)

def load_mp3_file(file_path: str | Path) -> AudioData:
    if isinstance(file_path, Path):
        file_path = str(file_path)

    audio_segment: AudioSegment = AudioSegment.from_mp3(file_path)
    sample_width = audio_segment.sample_width
    frame_bytes = audio_segment.raw_data
    
    audio_data = AudioData(
        frame_bytes,
        format=AudioData.sample_width_to_format(sample_width),
        channels=audio_segment.channels,
        rate=audio_segment.frame_rate
    )

    return audio_data

def save_mp3_file(audio_data: AudioData | AudioBuffer | AudioSegment, filename: str | Path) -> None:
    if isinstance(filename, str):
        filename = Path(filename)

    if not filename.parent.exists():
        filename.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(audio_data, AudioData):
        frames = audio_data.as_bytes()
        sample_rate = audio_data.rate
        format = audio_data.format
        channels = audio_data.channels
        sample_size, _ = audio_data.get_type_info()
        recording = AudioSegment(
            data=frames,
            sample_width=sample_size,
            frame_rate=sample_rate,
            channels=channels
        )

    elif isinstance(audio_data, AudioBuffer):
        frames = audio_data.get_bytes()
        sample_rate = audio_data.get_sample_rate()
        format = audio_data.get_audio_format()
        channels = audio_data.get_channels()
        sample_size = audio_data.get_sample_size()
        recording = AudioSegment(
            data=frames,
            sample_width=sample_size,
            frame_rate=sample_rate,
            channels=channels
        )
    elif isinstance(audio_data, AudioSegment):
        recording = audio_data
    
    else:
        raise ValueError(f"Unsupported audio data type: {type(audio_data)}")
    
    with open(filename, "wb") as mp3_file:
        recording.export(mp3_file, format="mp3")

async def save_mp3_file_async(audio_data: AudioData, filename: str | Path) -> None:
    await asyncio.to_thread(save_mp3_file, audio_data, filename)