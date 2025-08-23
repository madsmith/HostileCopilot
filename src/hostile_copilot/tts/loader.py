from hostile_copilot.config import OmegaConfig

from .engine import TTSEngine
from .engine_local import TTSEngineLocal
from .engine_openai import TTSEngineOpenAI


class TTSEngineLoader:
    def __init__(self, config: OmegaConfig):
        self._config = config

    def load(self) -> TTSEngine:
        engine_type = self._config.get("tts.engine")
        if engine_type == "local":
            return TTSEngineLocal(self._config)
        elif engine_type == "openai":
            return TTSEngineOpenAI(self._config)
        else:
            raise ValueError(f"Invalid TTS engine type: {engine_type}")