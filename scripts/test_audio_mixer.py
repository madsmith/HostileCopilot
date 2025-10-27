from pathlib import Path

from hostile_copilot.audio.files import load_wave_file, save_wave_file
from hostile_copilot.audio.audio_data import AudioData
from hostile_copilot.audio.mixer import ChannelMixer


def downsample_44_to_16(in_path: Path, out_path: Path) -> AudioData:
    """
    Load a 44.1 kHz WAV and resample to 16 kHz, preserving channels.
    Saves to out_path and returns the resampled AudioData.
    """
    audio: AudioData = load_wave_file(in_path)
    print(f"[Downsample] Loaded: {in_path} | rate={audio.rate} ch={audio.channels} frames={len(audio)}")
    audio_16k = audio.resample(16000)
    print(f"[Downsample] Resampled: rate={audio_16k.rate} ch={audio_16k.channels} frames={len(audio_16k)}")
    save_wave_file(audio_16k, out_path)
    print(f"[Downsample] Saved: {out_path}")
    return audio_16k


def upsample_16_to_44(in_path: Path, out_path: Path) -> AudioData:
    """
    Load a 16 kHz WAV and resample to 44.1 kHz, preserving channels.
    Saves to out_path and returns the resampled AudioData.
    """
    audio: AudioData = load_wave_file(in_path)
    print(f"[Upsample] Loaded: {in_path} | rate={audio.rate} ch={audio.channels} frames={len(audio)}")
    audio_44k = audio.resample(44100)
    print(f"[Upsample] Resampled: rate={audio_44k.rate} ch={audio_44k.channels} frames={len(audio_44k)}")
    save_wave_file(audio_44k, out_path)
    print(f"[Upsample] Saved: {out_path}")
    return audio_44k


def stereo44_to_mono16(in_path: Path, out_path: Path) -> AudioData:
    """
    Convert a stereo 44.1 kHz WAV to mono 16 kHz.
    - Downmix channels 2 -> 1 using ChannelMixer
    - Resample sample rate 44100 -> 16000
    Saves to out_path and returns the converted AudioData.
    """
    audio: AudioData = load_wave_file(in_path)
    print(f"[Stereo->Mono] Loaded: {in_path} | rate={audio.rate} ch={audio.channels} frames={len(audio)}")

    # Channel downmix to mono
    mixer = ChannelMixer(audio.channels, 1)
    sample_size = audio.get_sample_size()
    mono_bytes = mixer.mix(audio.as_bytes(), sample_size)
    mono_audio = AudioData(mono_bytes, format=audio.format, channels=1, rate=audio.rate)

    # Resample to 16k
    mono_16k = mono_audio.resample(16000)
    print(f"[Stereo->Mono] Resampled: rate={mono_16k.rate} ch={mono_16k.channels} frames={len(mono_16k)}")
    save_wave_file(mono_16k, out_path)
    print(f"[Stereo->Mono] Saved: {out_path}")
    return mono_16k


def mono44_to_stereo16(in_path: Path, out_path: Path) -> AudioData:
    """
    Convert a mono 44.1 kHz WAV to stereo 16 kHz.
    - Resample sample rate 44100 -> 16000
    - Upmix channels 1 -> 2 using ChannelMixer (duplicates into L/R)
    Saves to out_path and returns the converted AudioData.
    """
    audio: AudioData = load_wave_file(in_path)
    print(f"[Mono->Stereo] Loaded: {in_path} | rate={audio.rate} ch={audio.channels} frames={len(audio)}")

    # Resample to 16k first
    audio_16k = audio.resample(16000)

    # Upmix to stereo
    mixer = ChannelMixer(audio_16k.channels, 2)
    sample_size = audio_16k.get_sample_size()
    stereo_bytes = mixer.mix(audio_16k.as_bytes(), sample_size)
    stereo_16k = AudioData(stereo_bytes, format=audio_16k.format, channels=2, rate=audio_16k.rate)

    print(f"[Mono->Stereo] Converted: rate={stereo_16k.rate} ch={stereo_16k.channels} frames={len(stereo_16k)}")
    save_wave_file(stereo_16k, out_path)
    print(f"[Mono->Stereo] Saved: {out_path}")
    return stereo_16k

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    test_dir = project_root / "test_files"

    # Test 1: 44.1 kHz -> 16kHz
    in_44 = test_dir / "stereo_44100.wav"
    out_16 = test_dir / "stereo_44100_resampled_16000.wav"
    downsample_44_to_16(in_44, out_16)

    # Test 2: 16kHz -> 44.1kHz
    preferred_16_src = test_dir / "stereo_16000.wav"
    in_16 = preferred_16_src if preferred_16_src.exists() else out_16
    out_44 = test_dir / "stereo_16000_resampled_44100.wav"
    upsample_16_to_44(in_16, out_44)

    # Test 3: Stereo 44.1kHz -> Mono 16kHz
    in_stereo_44 = test_dir / "stereo_44100.wav"
    out_mono_16 = test_dir / "stereo_44100_to_mono_16000.wav"
    stereo44_to_mono16(in_stereo_44, out_mono_16)

    # Test 4: Mono 44.1kHz -> Stereo 16kHz
    in_mono_44 = test_dir / "mono_44100.wav"
    out_stereo_16 = test_dir / "mono_44100_to_stereo_16000.wav"
    mono44_to_stereo16(in_mono_44, out_stereo_16)


if __name__ == "__main__":
    main()