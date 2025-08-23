import asyncio
import logging
import pyaudio

from hostile_copilot.config import OmegaConfig
from hostile_copilot.tts.engine import TTSEngine
from hostile_copilot.audio import AudioDevice, load_wave_file, save_wave_file

from .voice import VoiceClient
from .tasks import Task, GetScreenBoundingBoxTask

logger = logging.getLogger(__name__)

class HostileCoPilotApp:
    def __init__(self, config: OmegaConfig):
        self._config = config

        self._audio_device: AudioDevice = AudioDevice(
            format=pyaudio.paInt16,
            rate=16000,
            channels=1,
            chunk_size=1024
        )

        self._voice_client: VoiceClient = VoiceClient(self._config, self._audio_device)
        self._voice_task: asyncio.Task | None = None

        self._is_running: bool = False

    async def initialize(self):
        logger.info("Initializing HostileCoPilotApp...")

        logger.info("Initializing AudioDevice...")
        self._audio_device.initialize()

        logger.info("Initializing VoiceClient...")
        await self._voice_client.initialize()

        self._voice_client.on_prompt(self._on_prompt)
        self._voice_client.on_immediate_activation(self._on_immediate_activation)

        self._audio_device.start()

    async def run(self):
        self._is_running = True

        try:
            self._voice_task = asyncio.create_task(
                self._voice_client.run(),
                name="HostileCoPilotApp::VoiceClient"
            )
        except asyncio.CancelledError:
            logger.info("VoiceClient task cancelled")
            self._is_running = False
            return

        # audio = await self._tts_engine.infer("Hello, how are you?")
        # print(f"Generated {len(audio)} bytes of audio.")

        # audio = audio.resample(16000)
        # self._audio_device.play(audio)

        # print(f"Sleeping for {audio.duration()} seconds")
        # await asyncio.sleep(audio.duration())
        # print("Audio should be done")

        # audio = load_wave_file("resources/stereo_test.wav")
        # print(f"Loaded wave file: {audio.rate} Hz, {audio.channels} channels, {len(audio)} bytes")
        # new_audio = audio.resample(16000)
        # self._audio_device.play(new_audio)

        # print(f"Sleeping for {new_audio.duration()} seconds")
        # await asyncio.sleep(new_audio.duration())
        # print("Audio should be done")
        
        # save_wave_file(new_audio, "resources/stereo_test_resampled.wav")

        while self._is_running:
            try:
                await asyncio.sleep(1)
            except KeyboardInterrupt:
                self._is_running = False


        await self._voice_task

    def _on_prompt(self, prompt: str):
        print(f"Prompt: {prompt}")

    async def _on_immediate_activation(self, wake_word: str):
        logger.info(f"Immediate activation: {wake_word}")
        if wake_word == "scan_this":
            task = GetScreenBoundingBoxTask(self._config)
            await task.run()
            print(f"Calibrated screen: {task._start} to {task._end}")
