
from typing import Protocol
from typing import Any

from hostile_copilot.audio import AudioData

class STTEngine(Protocol):
    def initialize(self): ...
    def infer(self, audio: AudioData, **inference_params: dict[str, Any]) -> str: ...