import asyncio

from hostile_copilot.config import OmegaConfig
from hostile_copilot.tts.engine import TTSEngine

class HostileCoPilotApp:
    def __init__(self, config: OmegaConfig):
        self._config = config

        model_id = self._config.get("tts.model_id")
        voices = self._config.get("tts.voices")
        self._tts_engine: TTSEngine = TTSEngine(model_id, voices)

        self._is_running: bool = False


    async def initialize(self):
        await self._tts_engine.initialize()

    async def run(self):
        audio = await self._tts_engine.infer("Hello, how are you?")
        print(f"Generated {len(audio)} bytes of audio.")
        while self._is_running:
            try:
                await asyncio.sleep(1)
            except KeyboardInterrupt:
                self._is_running = False
        pass