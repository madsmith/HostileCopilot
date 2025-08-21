import asyncio
import logging
import pyaudio

from hostile_copilot.config import OmegaConfig
from hostile_copilot.tts.engine import TTSEngine
from hostile_copilot.audio import AudioDevice

logger = logging.getLogger(__name__)

class HostileCoPilotApp:
    def __init__(self, config: OmegaConfig):
        self._config = config

        model_id = self._config.get("tts.model_id")
        voices = self._config.get("tts.voices")
        self._tts_engine: TTSEngine = TTSEngine(model_id, voices)

        self._audio_device: AudioDevice = AudioDevice(
            format=pyaudio.paInt16,
            rate=16000,
            channels=1,
            chunk_size=1024
        )

        self._is_running: bool = False


    async def initialize(self):
        logger.info("Initializing HostileCoPilotApp...")
        logger.info("Initializing TTS engine...")
        await self._tts_engine.initialize()

        logger.info("Initializing AudioDevice...")
        self._audio_device.initialize()
        self._audio_device.start()

    async def run(self):
        self._is_running = True
        
        audio = await self._tts_engine.infer("Hello, how are you?")
        print(f"Generated {len(audio)} bytes of audio.")

        self._audio_device.play(audio)

        while self._is_running:
            try:
                await asyncio.sleep(1)
            except KeyboardInterrupt:
                self._is_running = False
        pass