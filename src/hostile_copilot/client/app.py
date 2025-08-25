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
from hostile_copilot.client.uexcorp import UEXCorpClient
from hostile_copilot.mining_logger import MiningLogger

from .commodity_grader import CommodityGrader
from .voice import VoiceClient
from .tasks import (
    Task,
    GetScreenBoundingBoxTask,
    GetScreenLocationTask,
    MacroTask,
    MiningScanTask,
    MiningScanGraderTask
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

        self._uexcorp_client: UEXCorpClient = UEXCorpClient(self._config)

        self._mining_logger: MiningLogger = MiningLogger(self._config)
        self._commodity_grader: CommodityGrader = CommodityGrader(self._config)

        self._keyboard = Keyboard()

        self._agent: Agent | None = None
        self._calibration_agent: Agent | None = None

        self._is_running: bool = False

    async def initialize(self):
        logger.info("Initializing HostileCoPilotApp...")

        logger.info("Initializing AudioDevice...")
        self._audio_device.initialize()

        logger.info("Initializing VoiceClient...")
        await self._voice_client.initialize()

        logger.info("Initializing MiningLogger...")
        self._mining_logger.initialize()

        logger.info("Initializing CommodityGrader...")
        await self._commodity_grader.initialize(self._uexcorp_client)

        logger.info("Initializing Keyboard...")
        await self._keyboard.start()

        logger.info("Initializing Agent...")
        await self.initialize_agents()

        self._voice_client.on_prompt(self._on_prompt)
        self._voice_client.on_immediate_activation(self._on_immediate_activation)

        self._audio_device.start()

    async def initialize_agents(self):
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
            self._tool_perform_scan,
        ])

        calibration_toolset = FunctionToolset(tools=[
            self._tool_setup_mining_calibration,
            self._tool_setup_ping_scan_calibration,
            self._tool_setup_nav_search_calibration,
        ])

        # TODO: do we support agents from multiple models/providers?
        self._agent = Agent(
            model=model,
            system_prompt=system_prompt,
            toolsets=[toolset],
        )

        self._calibration_agent = Agent(
            model=model,
            system_prompt=system_prompt,
            toolsets=[calibration_toolset],
        )
    
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

        self._mining_logger.shutdown()

        await self._keyboard.stop()

    async def _on_prompt(self, wake_word: str, prompt: str):
        assert self._agent is not None
        response_text = None
        if wake_word == "mining_logger":
            response = await self._agent.run(prompt)

            response_text = response.output
        elif wake_word == "calibrate_system":
            response = await self._calibration_agent.run(prompt)

            response_text = response.output

        if response_text is not None:
            print(f"Prompt: {prompt}")
            print(f"Response: {response_text}")
            await self._voice_client.speak(response_text)

    async def _on_immediate_activation(self, wake_word: str):
        logger.info(f"Immediate activation: {wake_word}")
        if wake_word == "scan_this":
            await self._tool_perform_scan()
            
            # location = (3186, 426)
            
            # task = MacroTask(self._config, self._keyboard)
            # task.set_macro([
            #     ("sleep", 2),
            #     ("click", location),
            #     ("vkbd:press", "f2"),
            #     ("vkbd:sequence", list("hello"), {"interkey_delay": 0.02}),
            # ])
            # await task.run()
            
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

    async def _tool_perform_scan(self):
        """
        Perform a scan
        """
        task = MiningScanTask(self._config, self._app_config)
        await task.run()

        scan_result = task.scan_result
        if scan_result is None or scan_result.scan_data is None:
            await self._voice_client.speak("No scan data found")
            return

        self._mining_logger.log(scan_result.scan_data)

        grade_task = MiningScanGraderTask(
            self._config,
            self._app_config,
            scan_result,
            self._voice_client,
            self._commodity_grader
        )
        await grade_task.run()

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

    async def _tool_setup_ping_scan_calibration(self):
        task = GetScreenBoundingBoxTask(self._config)
        await task.run()
        
        start, end = task.bounding_box

        # Ensure path exists... TODO: make config support defaulting path parents
        if "calibration" not in self._app_config:
            self._app_config.set("calibration", {})
        if "calibration.ping_scan" not in self._app_config:
            self._app_config.set("calibration.ping_scan", {})
        
        self._app_config.set("calibration.ping_scan.start_x", start.x)
        self._app_config.set("calibration.ping_scan.start_y", start.y)
        self._app_config.set("calibration.ping_scan.end_x", end.x)
        self._app_config.set("calibration.ping_scan.end_y", end.y)

        omegaconf.OmegaConf.save(self._app_config._config, self._app_config_path)

    async def _tool_setup_nav_search_calibration(self):
        task = GetScreenLocationTask(self._config)
        await task.run()
        
        location = task.last_click

        # Ensure path exists... TODO: make config support defaulting path parents
        if "calibration" not in self._app_config:
            self._app_config.set("calibration", {})
        if "calibration.nav_search" not in self._app_config:
            self._app_config.set("calibration.nav_search", {})
        
        self._app_config.set("calibration.nav_search.location.x", location.x)
        self._app_config.set("calibration.nav_search.location.y", location.y)

        omegaconf.OmegaConf.save(self._app_config._config, self._app_config_path)