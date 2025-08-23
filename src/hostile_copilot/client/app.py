import asyncio
import logging
import omegaconf
from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
import pyaudio

from hostile_copilot.config import OmegaConfig, load_config
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

        app_config_path = self._config.get("app.config", "config/settings.yaml")
        self._app_config_path = Path(app_config_path)

        if not self._app_config_path.exists():
            self._app_config_path.touch()

        self._app_config = load_config(self._app_config_path)

        self._audio_device: AudioDevice = AudioDevice(
            format=pyaudio.paInt16,
            rate=16000,
            channels=1,
            chunk_size=1024
        )

        self._voice_client: VoiceClient = VoiceClient(self._config, self._audio_device)
        self._voice_task: asyncio.Task | None = None

        self._keyboard = Keyboard()

        self._agent: Agent | None = None

        self._is_running: bool = False

    async def initialize(self):
        logger.info("Initializing HostileCoPilotApp...")

        logger.info("Initializing AudioDevice...")
        self._audio_device.initialize()

        logger.info("Initializing VoiceClient...")
        await self._voice_client.initialize()

        logger.info("Initializing Keyboard...")
        await self._keyboard.start()

        logger.info("Initializing Agent...")
        await self.initialize_agent()

        self._voice_client.on_prompt(self._on_prompt)
        self._voice_client.on_immediate_activation(self._on_immediate_activation)

        self._audio_device.start()

    async def initialize_agent(self):
        api_key = self._config.get("agent.api_key", "")
        base_url = self._config.get("agent.base_url", None)
        provider = OpenAIProvider(api_key=api_key, base_url=base_url)

        model_id = self._config.get("agent.model_id")
        system_prompt = self._config.get("agent.system_prompt", "")
        model = OpenAIModel(
            provider=provider,
            model_name=model_id,
        )

        # Tool registration
        toolset = FunctionToolset(tools=[
            self._tool_perform_test,
        ])

        calibration_toolset = FunctionToolset(tools=[
            self._tool_setup_mining_calibration,
        ])

        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            toolsets=[toolset, calibration_toolset],

        )
        self._agent = agent
    
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

    async def _on_prompt(self, prompt: str):
        assert self._agent is not None
        print(f"Prompt: {prompt}")

        response = await self._agent.run(prompt)

        response_text = response.output
        print(f"Response: {response_text}")

        await self._voice_client.speak(response_text)

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

    async def _tool_perform_test(self):
        """
        Perform a test
        """
        print("Performing test")
        await self._voice_client.speak("Test")

    async def _tool_setup_mining_calibration(self):
        """
        One time task to calibrate the game screen UI coordinates for extracting
        mining data.
        """
        task = GetScreenBoundingBoxTask(self._config)
        await task.run()
        
        start, end = task.bounding_box

        # Ensure path exists... TODO: make config support defaulting path parents
        if "calibration" not in self._app_config:
            self._app_config.set("calibration", {})
        if "calibration.mining_scan" not in self._app_config:
            self._app_config.set("calibration.mining_scan", {})
        
        self._app_config.set("calibration.mining_scan.start_x", start.x)
        self._app_config.set("calibration.mining_scan.start_y", start.y)
        self._app_config.set("calibration.mining_scan.end_x", end.x)
        self._app_config.set("calibration.mining_scan.end_y", end.y)

        omegaconf.OmegaConf.save(self._app_config._config, self._app_config_path)