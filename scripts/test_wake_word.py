import sys
import signal
import pyaudio
import numpy as np
import torch

import openwakeword
from openwakeword import Model as OpenWakeWordModel
import silero_vad

from hostile_copilot.config import load_config

RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
VAD_CHUNK = 512             # VAD chunk frames (matches voice_client)
WAKEWORD_CHUNK = 1280       # OpenWakeWord expects ~0.08s windows
VAD_THRESHOLD = 0.5         # Same as voice_client

def to_torch_audio(int16_bytes: bytes, dtype=np.int16) -> torch.Tensor:
    """Convert raw int16 PCM bytes to torch tensor normalized to [-1, 1]."""
    np_audio = np.frombuffer(int16_bytes, dtype=dtype)
    # Normalize to float32 in [-1, 1]
    f32 = np_audio.astype(np.float32) / 32768.0
    return torch.from_numpy(f32)

def main():
    # Load config
    config = load_config()

    # Prepare openwakeword model list like voice_client
    model_entries = config.get("openwakeword.models")
    if not model_entries:
        print("Missing config 'openwakeword.models'", file=sys.stderr)
        sys.exit(1)

    inference_framework = config.get("openwakeword.inference_framework")
    if inference_framework not in ("onnx", "tflite"):
        print("Missing/invalid 'openwakeword.inference_framework' (onnx|tflite)", file=sys.stderr)
        sys.exit(1)

    models: list[str] = [
        (m.get("path") if "path" in m else m.get("name"))
        for m in model_entries
        if (m.get("path") or m.get("name"))
    ]
    if not models:
        print("No valid models found in 'openwakeword.models'", file=sys.stderr)
        sys.exit(1)

    # Initialize wake word and VAD
    openwakeword.utils.download_models()
    oww = OpenWakeWordModel(
        wakeword_models=models,
        inference_framework=inference_framework,
        enable_speex_noise_suppression=False,
    )
    vad = silero_vad.load_silero_vad(onnx=True)
    if vad is None:
        print("Failed to load Silero VAD", file=sys.stderr)
        sys.exit(1)

    # Setup PyAudio input stream
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=VAD_CHUNK,
    )

    # Graceful shutdown
    running = True
    def _sigint(_sig, _frm):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _sigint)

    # Separate buffers for VAD and Wake Word
    vad_buf = bytearray()
    wake_buf = bytearray()

    # Tracking consecutive determinations
    # Last VAD state and consecutive determination count
    last_speech: bool | None = None
    last_vad_score: float = 0.0
    speech_count: int = 0

    last_wake: tuple[str, ...] | None = None
    wake_count: int = 0

    prev_line_len: int = 0

    # Open log file to persist each line for later review
    log_path = "test_wake_word.log"
    with open(log_path, "a", encoding="utf-8") as log_fh:
        print("Listening... Ctrl+C to stop.")

        try:
            # Main processing loop
            while running:
                # Determine how many frames we need to satisfy the next VAD and/or OWW window
                vad_frames = len(vad_buf) // 2
                oww_frames = len(wake_buf) // 2

                vad_needed = max(0, VAD_CHUNK - vad_frames)
                oww_needed = max(0, WAKEWORD_CHUNK - oww_frames)

                frames_to_read = 0
                if vad_needed == 0 or oww_needed == 0:
                    # At least one check can run now; don't block on read
                    frames_to_read = 0
                else:
                    # Read the minimum to fulfill either check
                    frames_to_read = min(vad_needed, oww_needed)

                if frames_to_read > 0:
                    data = stream.read(frames_to_read, exception_on_overflow=False)
                    # Append new audio to both buffers
                    vad_buf.extend(data)
                    wake_buf.extend(data)

                # Run VAD if we have enough frames
                vad_ran = False
                if (len(vad_buf) // 2) >= VAD_CHUNK:
                    vad_window = vad_buf[: VAD_CHUNK * 2]
                    del vad_buf[: VAD_CHUNK * 2]

                    torch_audio = to_torch_audio(vad_window)
                    last_vad_score = vad(torch_audio, RATE).item()
                    is_speech = last_vad_score > VAD_THRESHOLD
                    vad_ran = True

                    if last_speech is None or last_speech != is_speech:
                        speech_count = 1
                    else:
                        speech_count += 1
                    last_speech = is_speech

                # Run Wake Word detection if we have enough frames
                oww_ran = False
                detected_words: list[str] | None = None
                if (len(wake_buf) // 2) >= WAKEWORD_CHUNK:
                    window = wake_buf[: WAKEWORD_CHUNK * 2]
                    del wake_buf[: WAKEWORD_CHUNK * 2]

                    np_window = np.frombuffer(window, dtype=np.int16)
                    detection = oww.predict(np_window)  # dict[str,float]
                    activation_threshold = config.get("voice.activation_threshold", 0.5)
                    words = [k for k, v in detection.items() if v > activation_threshold]
                    if words:
                        detected_words = words
                    oww_ran = True

                    # Update consecutive wake-word determination counts
                    new_wake_key: tuple[str, ...] | None = tuple(sorted(detected_words)) if detected_words else None
                    if last_wake is None or last_wake != new_wake_key:
                        wake_count = 1
                    else:
                        wake_count += 1
                    last_wake = new_wake_key

                # If a check didn't run, retain previous state and do not change counts
                is_speech_output = bool(last_speech) if last_speech is not None else False
                vad_score_output = last_vad_score
                display_words = (
                    ", ".join(last_wake) if isinstance(last_wake, tuple) and last_wake is not None else "None"
                )

                # Build single-line status
                line = (
                    f"Speech: {is_speech_output:<5}  "
                    f"Score: {vad_score_output:.3f}  "
                    f"Count: {speech_count:<5}  "
                    f"Wake Words: {display_words:<30}  "
                    f"Count: {wake_count:<5}"
                )

                # Write to log as a distinct line
                try:
                    log_fh.write(line + "\n")
                    log_fh.flush()
                except Exception:
                    pass

                # Refresh same console line; pad to clear leftovers
                padded = line + (' ' * max(0, prev_line_len - len(line)))
                sys.stdout.write('\r' + padded)
                sys.stdout.flush()
                prev_line_len = len(line)

        finally:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            pa.terminate()
            # Move to next line after loop ends
            print()

if __name__ == "__main__":
    main()