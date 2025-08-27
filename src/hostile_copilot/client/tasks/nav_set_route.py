

from hostile_copilot.utils.input.keyboard import Keyboard
from hostile_copilot.config import OmegaConfig
from hostile_copilot.client.voice import VoiceClient

from .base import Task
from .macro import MacroTask

class NavSetRouteTask(Task):
    def __init__(
        self,
        config: OmegaConfig,
        app_config: OmegaConfig,
        voice_client: VoiceClient,
        keyboard: Keyboard,
        destination: str
    ):
        super().__init__(config)
        self._app_config = app_config
        self._voice_client = voice_client
        self._keyboard = keyboard
        self._destination: str = destination

    async def run(self):
        search_location = self._app_config.get("calibration.nav_locations.search")
        first_location = self._app_config.get("calibration.nav_locations.first_result")
        route_location = self._app_config.get("calibration.nav_locations.route")

        if search_location is None or first_location is None or route_location is None:
            await self._voice_client.speak("Missing calibration inputs for navigation route.")
            return
        
        task = MacroTask(self._config, self._keyboard)
        task.set_macro([
            ("vkbd:press", "f2"),
            ("vkbd:sleep", 2.2),
            ("click", (search_location.x, search_location.y)),
            ("vkbd:sleep", 0.2),
            ("click", (search_location.x, search_location.y)),
            ("vkbd:sequence", list(self._destination), {"interkey_delay": 0.02, "press_duration": 0.1}),
            ("vkbd:sleep", 0.5),
            ("click", (first_location.x, first_location.y)),
            ("vkbd:sleep", 0.5),
            ("click", (route_location.x, route_location.y)),
        ])
        await task.run()
        
        
