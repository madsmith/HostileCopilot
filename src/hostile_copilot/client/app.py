import asyncio
import logging
import pyaudio

from hostile_copilot.config import OmegaConfig
from hostile_copilot.tts.engine import TTSEngine
from hostile_copilot.audio import AudioDevice
from hostile_copilot.audio.files import load_wave_file, save_wave_file

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

        audio = audio.resample(16000)
        self._audio_device.play(audio)

        print(f"Sleeping for {audio.duration()} seconds")
        await asyncio.sleep(audio.duration())
        print("Audio should be done")

        audio = load_wave_file("resources/stereo_test.wav")
        print(f"Loaded wave file: {audio.rate} Hz, {audio.channels} channels, {len(audio)} bytes")
        new_audio = audio.resample(16000)
        self._audio_device.play(new_audio)

        print(f"Sleeping for {new_audio.duration()} seconds")
        await asyncio.sleep(new_audio.duration())
        print("Audio should be done")
        
        save_wave_file(new_audio, "resources/stereo_test_resampled.wav")

        while self._is_running:
            try:
                await asyncio.sleep(1)
                break
            except KeyboardInterrupt:
                self._is_running = False
        pass