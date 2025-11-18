import anyio
import asyncio
import logging
import omegaconf
from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
import pyaudio
import time
from typing import Any

from hostile_copilot.client.tasks.ping_analysis import PingAnalysisResult
from hostile_copilot.config import OmegaConfig, load_config
from hostile_copilot.audio import AudioDevice
from hostile_copilot.utils.input.keyboard import Keyboard
from hostile_copilot.utils.speech import roundify_numbers
from hostile_copilot.client.uexcorp import UEXCorpClient
from hostile_copilot.client.regolith import RegolithClient
from hostile_copilot.mining_logger import MiningLogger
from hostile_copilot.client.components.locations import LocationProvider

from .commodity_grader import CommodityGrader
from .voice import VoiceClient
from .tasks import (
    Task,
    GetScreenBoundingBoxTask,
    GetScreenLocationTask,
    MacroTask,
    MiningScanTask,
    MiningScanGraderTask,
    NavSetRouteTask,
    SetLocationResponse,
    PingAnalysisTask,
    PingResponse
)
from .tasks.types import CommodityData, ScanResponse, NavSetRouteResponse

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
        self._regolith_client: RegolithClient = RegolithClient(self._config)

        self._mining_logger: MiningLogger = MiningLogger(self._config)
        self._commodity_grader: CommodityGrader = CommodityGrader(self._config)
        self._location_provider: LocationProvider = LocationProvider(self._config, self._uexcorp_client, self._regolith_client)

        self._keyboard = Keyboard()

        self._agent: Agent | None = None
        self._calibration_agent: Agent | None = None
        self._calibration_lock: asyncio.Lock = asyncio.Lock()
        self._copilot_agent: Agent | None = None

        self._is_running: bool = False

        # State
        self._current_location: str | None = None
        self._agent_prompt_lock: asyncio.Lock = asyncio.Lock()
        self._agent_prompt_future: asyncio.Future | None = None
        self._prompt_tool_calls: int = 0
        self._last_prompt_time: float = 0.0


    async def initialize(self, listen: bool = True):
        logger.info("Initializing HostileCoPilotApp...")

        if listen:
            logger.info("Initializing AudioDevice...")
            self._audio_device.initialize()

            logger.info("Initializing VoiceClient...")
            await self._voice_client.initialize()

        logger.info("Initializing MiningLogger...")
        self._mining_logger.initialize()

        logger.info("Initializing CommodityGrader...")
        await self._commodity_grader.initialize(self._uexcorp_client)
        self._commodity_grader.save_tier_map("tiermap.json")

        logger.info("Initializing Keyboard...")
        await self._keyboard.start()

        logger.info("Initializing Agent...")
        await self.initialize_agents()

        if listen:
            self._voice_client.on_prompt(self._on_prompt)
            self._voice_client.on_immediate_activation(self._on_immediate_activation)

            self._audio_device.start()

        await self._load_gravity_well()
            
        logger.info("HostileCoPilotApp initialized")

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

        copilot_model_id = self._config.get("copilot.model_id")
        copilot_system_prompt = self._config.get("copilot.system_prompt", "")
        copilot_model = OpenAIModel(
            provider=provider,
            model_name=copilot_model_id,
        )

        # Tool registration
        toolset = FunctionToolset(tools=[
            self._tool_search_gravity_well_locations,
            self._tool_set_gravity_well_location,
            self._tool_search_navigation_locations,
            self._prompt_user_for_input,
            self._tool_get_commodity_data,
        ])

        copilot_toolset = FunctionToolset(tools=[
            self._prompt_user_for_input,
            self._tool_set_gravity_well_location,
            self._tool_search_gravity_well_locations,
            self._tool_search_navigation_locations,
            self._tool_set_navigation_route,
            self._tool_get_commodity_data,
        ])

        calibration_toolset = FunctionToolset(tools=[
            self._tool_setup_mining_calibration,
            self._tool_setup_ping_scan_calibration,
            self._tool_setup_nav_route_calibration,
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

        self._copilot_agent = Agent(
            model=copilot_model,
            system_prompt=copilot_system_prompt,
            toolsets=[copilot_toolset],
        )

    async def _load_gravity_well(self):
        gravity_well = self._app_config.get("gravity_well", "")
        if gravity_well:
            max_location_age = self._config.get("app.max_location_age", 3600)

            location = gravity_well.get("location", "")
            timestamp = gravity_well.get("timestamp", 0)
            
            if time.time() - timestamp < max_location_age:
                logger.info(f"Loading gravity well location: {location}")
                s
                self._current_location = location
    
    async def run(self):
        self._is_running = True
        await self._voice_client.speak("System is online")

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
        # TODO: maybe have prompt take in audio and app can decide to do STT
        # or use audio as prompt inference inputs
        assert self._agent is not None
        response_text = None
        if wake_word == "mining_logger":
            response = await self._agent.run(prompt)

            response_text = response.output
        elif wake_word == "calibrate_system":
            response = await self._calibration_agent.run(prompt)

            response_text = response.output
        elif wake_word == "copilot":
            response = await self._copilot_agent.run(prompt)

            response_text = response.output

        elif wake_word == "agent_prompt":
            if self._agent_prompt_future is not None and not self._agent_prompt_future.done():
                cleaned = prompt.strip()
                norm = cleaned.lower()
                cancel_phrases = {"cancel", "nevermind", "never mind", "stop", "abort", "exit"}
                if norm in cancel_phrases:
                    self._agent_prompt_future.set_result("__CANCEL__")
                    return
                alnum = "".join(ch for ch in cleaned if ch.isalnum())
                min_chars = int(self._app_config.get("agent.stt_min_chars", 2))
                if len(alnum) < min_chars:
                    return
                self._agent_prompt_future.set_result(cleaned)
            else:
                logger.warning("Received agent_prompt but no pending prompt future")

        else:
            logger.warning(f"Wake word {wake_word} not recognized")

        if response_text is not None:
            print(f"Prompt: {prompt}")
            print(f"Response: {response_text}")
            spoken_response = roundify_numbers(response_text)
            await self._voice_client.speak(spoken_response)

    async def _on_immediate_activation(self, wake_word: str):
        logger.info(f"Immediate activation: {wake_word}")
        if wake_word == "scan_this":
            await self._tool_perform_scan(tool_mode=False)
        elif wake_word == "analyze_ping":
            await self._tool_perform_analysis()
        elif wake_word == "stop":
            self._voice_client.stop_playback()

    async def _prompt_user_for_input(self, prompt: str) -> str | None:
        """
        Prompt the user for input

        Args:
            prompt (str): The prompt to ask to the user

        Returns:
            str: The user's input
        """

        async with self._agent_prompt_lock:
            now = time.monotonic()
            cooldown = float(self._app_config.get("agent.prompt_cooldown", 10.0))
            if now - self._last_prompt_time > cooldown:
                self._prompt_tool_calls = 0
            self._last_prompt_time = now

            self._prompt_tool_calls += 1
            max_calls = int(self._app_config.get("agent.max_prompt_tool_calls", 3))
            if self._prompt_tool_calls > max_calls:
                await self._voice_client.speak("Cancelling prompt.", wait_for_completion=True)
                self._prompt_tool_calls = 0
                return None

            await self._voice_client.speak(prompt, wait_for_completion=True)

            self._agent_prompt_future = asyncio.get_event_loop().create_future()
            self._voice_client.start_recording(confirmed=True, wake_word="agent_prompt")

            try:
                with anyio.fail_after(self._app_config.get("agent.prompt_timeout", 10)):
                    await self._agent_prompt_future
                response = self._agent_prompt_future.result()
                if response in {"__CANCEL__", "you"} or response is None:
                    self._prompt_tool_calls = 0
                    return None
                self._prompt_tool_calls = 0
                return response
            except TimeoutError:
                await self._voice_client.speak("Failed trying to get response from user")
                self._prompt_tool_calls = 0
                raise RuntimeError("Failed trying to get response from user")
            finally:
                self._agent_prompt_future = None

    async def _tool_perform_scan(self, tool_mode = True) -> ScanResponse | str | None:
        """
        Perform a scan of the mineable object to determine the composition and grade
        of the object.
        """
        
        # if self._current_location is None:
        #     if tool_mode:
        #         return "Please specify the current location"
        #     else:
        #         await self._voice_client.speak("Please specify the current location")
        #         return
        
        logger.info("Performing scan...")
        task = MiningScanTask(
            self._config,
            self._app_config,
            self._commodity_grader)
        await task.run()
        logger.info("Scan complete")

        scan_result = task.scan_result
        if scan_result is None or scan_result.scan_data is None:
            if tool_mode:
                return "No scan data found"
            else:
                await self._voice_client.speak("No scan data found")
            return

        if self._current_location is not None:
            self._mining_logger.log(scan_result.scan_data)

        logger.debug(f"Scan Data: {scan_result.scan_data}")
        grade_task = MiningScanGraderTask(
            self._config,
            self._app_config,
            scan_result,
            self._voice_client,
            self._commodity_grader,
            tool_mode=tool_mode
        )
        await grade_task.run()

        return scan_result

    async def _tool_perform_analysis(self) -> PingResponse | None:
        """
        Perform an analysis of the current radar ping.
        """
        
        task = PingAnalysisTask(self._config, self._app_config)
        await task.run()

        ping_result = task.ping_result
        if ping_result is None or ping_result.ping_data is None:
            await self._voice_client.speak("No ping data found")
            return
        
        analysis_result = task.process_ping(ping_result.ping_data)
        if analysis_result is None:
            logger.warning("Unexpected empty response while processing ping")
            await self._voice_client.speak("No ping data found")
            return

        if isinstance(analysis_result, PingAnalysisResult):
            if analysis_result.count == 1:
                msg = f"Detected {analysis_result.name}"
            else:
                msg = f"Detected {analysis_result.name}. Quantity: {analysis_result.count}"
            await self._voice_client.speak(msg)
        else:
            logger.info(f"Signature readout: {analysis_result}")
        
        return ping_result

    async def _tool_set_gravity_well_location(self, location: str) -> SetLocationResponse:
        """
        Set the current location to the gravity well in which mining activity is occuring.

        Args:
            location (str): The name of the gravity well.
        """
        matches = await self._location_provider.search(location, gravity_well=True)

        if len(matches) == 1:
            self._current_location = matches[0].name

            # Save current location and time to settings
            self._app_config.set("gravity_well.location", self._current_location)
            self._app_config.set("gravity_well.timestamp", time.time())

            return SetLocationResponse(
                success=True,
                message=f"Current location set to {self._current_location}"
            )
        elif len(matches) > 1:
            return SetLocationResponse(
                success=False,
                message=f"Location {location} is ambiguous.  Matches: {', '.join([match.name for match in matches])}"
            )
        else:
            return SetLocationResponse(
                success=False,
                message=f"Location {location} not found."
            )
    
    async def _tool_search_gravity_well_locations(self, search_string: str) -> list[str]:
        """
        Search for gravity well locations that can be recorded for mining.

        Args:
            search_string (str): The search string to use.

        Returns:
            list[str]: A list of location names matching the search string.
        """
        locations = await self._location_provider.search(search_string, gravity_well=True)
        return [location.name for location in locations]
    
    async def _tool_search_navigation_locations(self, search_string: str) -> list[str]:
        """
        Search for locations that can be navigated to.  This is a distinct list separate from
        gravity well locations that can be recorded for mining.

        Args:
            search_string (str): The search string to use.

        Returns:
            list[str]: A list of location names matching the search string.
        """
        locations = await self._location_provider.search(search_string, navigable=True)
        return [location.name for location in locations]

    async def _tool_set_navigation_route(self, destination: str) -> NavSetRouteResponse:
        """
        Set a navigation route to the specified destination.
        """

        matches = await self._location_provider.search(destination, navigable=True)

        if len(matches) > 1:
            return NavSetRouteResponse(
                success=False,
                message=f"Location {destination} is ambiguous.  Matches: {', '.join([match.name for match in matches])}"
            )
        elif len(matches) == 0:
            return NavSetRouteResponse(
                success=False,
                message=f"Location {destination} not found."
            )
        else:
            task = NavSetRouteTask(self._config, self._app_config, self._voice_client, self._keyboard, self._location_provider, destination)
            return await task.run()

    async def _tool_get_commodity_data(self) -> list[CommodityData]:
        """
        Get commodity data, including buy/sell prices, legality status, and refined status.
        """
        commodities: list[dict[str, Any]] = await self._uexcorp_client.fetch_commodities()
        print(commodities)

        # convert into a list of pydantic models
        commodity_data: list[CommodityData] = [
            CommodityData(**commodity)
            for commodity in commodities
            if bool(commodity.get("is_visible"))
        ]

        return commodity_data

    async def _tool_setup_mining_calibration(self):
        """
        One time task to calibrate the game screen UI coordinates for extracting
        mining data.
        """
        async with self._calibration_lock:
            await self._voice_client.speak(
                "Select region of the screen with mining scan data.",
                wait_for_completion=True
            )
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
        """
        One time task to calibrate the game screen UI coordinates for extracting
        ping scan data.
        """
        async with self._calibration_lock:
            await self._voice_client.speak(
                "Select region of the screen with ping scan data.",
                wait_for_completion=True
            )
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

    async def _tool_setup_nav_route_calibration(self):
        """
        One time task to calibrate the game screen UI coordinates for searching
        and setting navigation routes.
        """
        async with self._calibration_lock:
            await self._voice_client.speak(
                "Click location of the navigation search button.",
                wait_for_completion=True
            )
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