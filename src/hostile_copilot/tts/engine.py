from typing import Protocol
from hostile_copilot.audio import AudioData
from typing import Any

class TTSEngine(Protocol):
    async def initialize(self): ...
    async def infer(self, text: str, inference_params: dict[str, Any] | None = None) -> AudioData: ...