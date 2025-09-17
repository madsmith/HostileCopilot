

from hostile_copilot.utils.input.keyboard import Keyboard
from hostile_copilot.config import OmegaConfig
from hostile_copilot.client.voice import VoiceClient
from hostile_copilot.client.components.locations import LocationProvider, LocationValidationResponse, LocationType

from .base import Task
from .macro import MacroTask
from .types import NavSetRouteResponse

class NavSetRouteTask(Task):
    def __init__(
        self,
        config: OmegaConfig,
        app_config: OmegaConfig,
        voice_client: VoiceClient,
        keyboard: Keyboard,
        location_provider: LocationProvider,
        destination: str
    ):
        super().__init__(config)
        self._app_config: OmegaConfig = app_config
        self._voice_client: VoiceClient = voice_client
        self._keyboard: Keyboard = keyboard
        self._location_provider: LocationProvider = location_provider
        self._destination: str = destination

    async def run(self) -> NavSetRouteResponse:
        search_location = self._app_config.get("calibration.nav_locations.search")
        first_location = self._app_config.get("calibration.nav_locations.first_result")
        route_location = self._app_config.get("calibration.nav_locations.route")

        if search_location is None or first_location is None or route_location is None:
            return NavSetRouteResponse(success=False, message="System requires calibration for navigation route.")

        # Attempt to resolve the location
        # Note caller already does this validation so it generally should not trigger
        matches = await self._location_provider.search(self._destination)
        valid_location: LocationType | None = None
        if len(matches) > 1:
            return NavSetRouteResponse(success=False, message=f"Location {self._destination} is ambiguous.\n\nMatches:\n{[location.name for location in matches]}. Prompt for clarification.")
        elif len(matches) == 0:
            return NavSetRouteResponse(success=False, message=f"Location {self._destination} not found.")
        else:
            valid_location = matches[0]

        location_stem = await self._location_provider.get_location_stem(valid_location.name)
        search_term = location_stem if location_stem is not None else self._destination
        
        task = MacroTask(self._config, self._keyboard)
        task.set_macro([
            ("vkbd:press", "f2"),
            ("vkbd:sleep", 2.2),
            ("click", (search_location.x, search_location.y)),
            ("vkbd:sleep", 0.2),
            ("click", (search_location.x, search_location.y)),
            ("vkbd:sequence", list(search_term), {"interkey_delay": 0.02, "press_duration": 0.1}),
            ("vkbd:sleep", 1),
            ("click", (first_location.x, first_location.y)),
            ("vkbd:sleep", 0.5),
            ("click", (route_location.x, route_location.y)),
        ])
        await task.run()
        
        return NavSetRouteResponse(success=True, message=f"Navigation route set to {valid_location.name}")
        
        
