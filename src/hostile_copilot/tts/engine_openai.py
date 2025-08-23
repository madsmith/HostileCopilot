import openai
import io
from typing import Any
import pyaudio
from pydub import AudioSegment

from hostile_copilot.audio import AudioData
from hostile_copilot.config import OmegaConfig


class TTSEngineOpenAI:
    def __init__(self, config: OmegaConfig):
        self._config = config
        api_key = config.get("openai.api_key")
        if api_key is None:
            raise ValueError("Missing config 'openai.api_key'")

        self._client = openai.OpenAI(api_key=api_key)

        self._model_id = config.get("tts-openai.model_id")
        self._voice = config.get("tts-openai.voice")
        self._instructions = config.get("tts-openai.instructions")

    async def initialize(self):
        pass

    async def infer(self, text: str, inference_params: dict[str, Any] | None = None) -> AudioData:
        params = {
            "model": self._model_id,
            "voice": self._voice,
            "input": text,
            "instructions": self._instructions,
        }
        params.update(inference_params or {})
        params["response_format"] = "mp3"

        response = self._client.audio.speech.create(**params)

        audio_bytes = response.content
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")

        return AudioData(audio.raw_data, format=pyaudio.paInt16, channels=audio.channels, rate=audio.frame_rate)