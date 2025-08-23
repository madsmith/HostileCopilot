import openai
from typing import Any
import os
from tempfile import NamedTemporaryFile

from hostile_copilot.config import OmegaConfig
from hostile_copilot.audio import AudioData
from hostile_copilot.audio import save_mp3_file

class STTEngine:
    def __init__(self, config: OmegaConfig):
        self._config: OmegaConfig = config
        api_key = config.get("openai.api_key")
        if api_key is None:
            raise ValueError("Missing config 'openai.api_key'")
        self._client = openai.Client(api_key=api_key)

    def initialize(self):
        pass

    def infer(self, audio: AudioData, **inference_params: dict[str, Any]) -> str:
        """Synchronously transcribe audio using OpenAI. Writes a temporary MP3 and cleans it up after use."""
        model_id = self._config.get("stt.model_id", "whisper-1")

        tmp: NamedTemporaryFile | None = None
        try:
            tmp = NamedTemporaryFile(delete=False, suffix=".mp3")
            tmp_path = tmp.name
            tmp.close()  # allow other processes to read the file on Windows

            # Save audio to temporary MP3
            save_mp3_file(audio, tmp_path)

            # Transcribe
            with open(tmp_path, "rb") as f:
                transcription = self._client.audio.transcriptions.create(
                    model=model_id,
                    file=f,
                )
            print(f"Transcribed in {end - start} seconds")
            
            # openai v1 returns an object with .text
            return getattr(transcription, "text", "")
        finally:
            if tmp is not None:
                try:
                    os.remove(tmp.name)
                except OSError:
                    pass