import asyncio
import logging
import pyaudio

from hostile_copilot.config import OmegaConfig
from hostile_copilot.audio import AudioDevice
from hostile_copilot.utils.input.keyboard import Keyboard

from .voice import VoiceClient
from .tasks import (
    Task,
    GetScreenBoundingBoxTask,
    GetScreenLocationTask,
    MacroTask
)

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

        self._keyboard = Keyboard()

        self._is_running: bool = False

    async def initialize(self):
        logger.info("Initializing HostileCoPilotApp...")

        logger.info("Initializing AudioDevice...")
        self._audio_device.initialize()

        logger.info("Initializing VoiceClient...")
        await self._voice_client.initialize()

        logger.info("Initializing Keyboard...")
        await self._keyboard.start()

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

        while self._is_running:
            try:
                await asyncio.sleep(.2)
            except KeyboardInterrupt:
                self._is_running = False


        if self._voice_task is not None:
            await self._voice_task

        await self._keyboard.stop()

    def _on_prompt(self, prompt: str):
        print(f"Prompt: {prompt}")

    async def _on_immediate_activation(self, wake_word: str):
        logger.info(f"Immediate activation: {wake_word}")
        if wake_word == "scan_this":
            location = (3186, 426)
            
            task = MacroTask(self._config, self._keyboard)
            task.set_macro([
                ("sleep", 2),
                ("click", location),
                ("vkbd:press", "f2"),
                ("vkbd:sequence", list("hello"), {"interkey_delay": 0.02}),
            ])
            await task.run()
            
            # task = GetScreenLocationTask(self._config)
            # await task.run()
            # print(f"Located click: {task.last_click}")
            
            # task = GetScreenBoundingBoxTask(self._config)
            # await task.run()
            # print(f"Calibrated screen: {task._start} to {task._end}")
