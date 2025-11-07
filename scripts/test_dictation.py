import argparse
import signal
import sys
import time
from typing import Optional

import numpy as np
import silero_vad

from hostile_copilot.config import load_config, OmegaConfig
from hostile_copilot.audio import AudioDevice, AudioData, numpy_to_tensor
from hostile_copilot.stt import STTEngineLoader, STTEngine


def run(config_path: Optional[str]):
    config: OmegaConfig = load_config(config_path)

    # Load STT engine
    stt_engine: STTEngine = STTEngineLoader(config).load()
    stt_engine.initialize()

    # Prepare audio (mic only)
    audio = AudioDevice()
    audio.set_playback_enabled(False)
    audio.initialize()
    audio.start()

    # Load Silero VAD (onnx)
    vad_model = silero_vad.load_silero_vad(onnx=True)
    if vad_model is None:
        print("Failed to load Silero VAD", file=sys.stderr)
        return 1

    sample_rate = audio.rate
    chunk_size = 512  # frames per VAD step (matches voice_client)

    vad_threshold = float(config.get("voice.vad_threshold", 0.5))
    speech_timeout_s = float(config.get("voice.speech_timeout", 1.5))
    min_duration_s = float(config.get("voice.min_recording_duration", 1.0))

    in_speech = False
    silence_for = 0.0
    speech_bytes = bytearray()

    print("Listening... Press Ctrl+C to stop.")

    def cleanup_and_exit(*_):
        try:
            audio.shutdown()
        finally:
            sys.exit(0)

    signal.signal(signal.SIGINT, cleanup_and_exit)

    try:
        while True:
            frames = audio.read(chunk_size)
            # Keep a copy for buffering when needed
            frames_bytes = frames.tobytes() if isinstance(frames, np.ndarray) else bytes(frames)

            # VAD expects a torch tensor in range [-1, 1], use provided helper
            torch_audio = numpy_to_tensor(frames, np.int16)
            vad_score = vad_model(torch_audio, sample_rate).item()
            is_speech = vad_score > vad_threshold

            if is_speech:
                if not in_speech:
                    # Start of speech
                    in_speech = True
                    speech_bytes.clear()
                    silence_for = 0.0
                speech_bytes.extend(frames_bytes)
            else:
                if in_speech:
                    silence_for += chunk_size / sample_rate
                    speech_bytes.extend(frames_bytes)

                    if silence_for >= speech_timeout_s:
                        # End of speech segment
                        in_speech = False
                        silence_for = 0.0

                        if len(speech_bytes) == 0:
                            continue

                        audio_data = AudioData(
                            data=bytes(speech_bytes),
                            format=audio.format,
                            channels=audio.channels,
                            rate=audio.rate,
                        )

                        if audio_data.duration() < min_duration_s:
                            # Too short, skip
                            speech_bytes.clear()
                            continue

                        try:
                            text = stt_engine.infer(audio_data)
                            print(text)
                        except Exception as e:
                            print(f"STT inference failed: {e}", file=sys.stderr)
                        finally:
                            speech_bytes.clear()
                # else: remain idle

            # Allow loop to be interruptible
            time.sleep(0.0)
    finally:
        audio.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple dictation test using Silero VAD and STT engine")
    parser.add_argument("--config", "-c", default=None, help="Path to configuration file")
    args = parser.parse_args()
    sys.exit(run(args.config))
