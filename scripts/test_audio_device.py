import logging
from pathlib import Path
import pyaudio
import time
import sys


# Ensure project's src/ is on sys.path for local execution
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
_src_path = _project_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from hostile_copilot.audio import AudioDevice, load_wave_file


def main() -> int:
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().setLevel("TRACE")

    # Resolve project root and wav path
    project_root = _project_root
    wav_path = project_root / "test_files" / "counting.wav"

    if not wav_path.exists():
        print(f"WAV file not found: {wav_path}", file=sys.stderr)
        return 1

    # Load WAV as AudioData to get proper format/rate/channels
    audio = load_wave_file(wav_path)

    # Configure AudioDevice to match the WAV to avoid resampling
    rate = 44100
    channels = 2
    print(f"Opening Audio Device: rate={rate}, channels={channels}")
    device = AudioDevice(
        format=pyaudio.paInt16,
        rate=rate,
        channels=channels,
        chunk_size=1024,
    )

    # We only need playback for this test
    device.set_mic_enabled(False)

    try:
        print("Initializing audio device...")
        device.initialize()
        device.start()

        print(f"Playing: {wav_path.name} ({audio.rate} Hz, {audio.channels} ch, duration {audio.duration():.2f}s)")
        device.play(audio)

        # Wait for audio to finish plus a small buffer
        print(f"Waiting for audio to finish: {audio.duration():.2f}s")
        time.sleep(audio.duration() + 0.5)
        print("Done.")
        return 0
    finally:
        device.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
