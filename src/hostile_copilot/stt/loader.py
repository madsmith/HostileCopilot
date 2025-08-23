from hostile_copilot.config import OmegaConfig

from .engine_openai import STTEngineOpenAI
from .engine_local import STTEngineLocal
from .engine import STTEngine

class STTEngineLoader:
    def __init__(self, config: OmegaConfig):
        self._config = config

    def load(self) -> STTEngine:
        engine_type = self._config.get("stt.engine")
        if engine_type == "openai":
            return STTEngineOpenAI(self._config)
        elif engine_type == "local":
            return STTEngineLocal(self._config)
        else:
            raise ValueError(f"Invalid STT engine type: {engine_type}")