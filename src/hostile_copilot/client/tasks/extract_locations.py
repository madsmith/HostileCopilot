import cv2
import logging
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel, Field
from pathlib import Path
from enum import Enum

from hostile_copilot.client.components.locations import LocationProvider, LocationInfo
from hostile_copilot.config import OmegaConfig

from .base import Task

logger = logging.getLogger(__name__)

class LocationType(Enum):
    PLANET = "Planet"
    MOON = "Moon"
    CITY = "City"
    SPACE_STATION = "Space Station"
    POINT_OF_INTEREST = "Point of Interest"

class LocationData(BaseModel):
    name: str
    type: LocationType

class LocationsResponse(BaseModel):
    locations: list[LocationData] = Field(
        default_factory=list,
        description="Flat list of location names/types extracted from the input. If none found, return an empty list.",
    )

class ExtractLocationsTask(Task):
    def __init__(
        self,
        config: OmegaConfig,
        app_config: OmegaConfig,
        location_provider: LocationProvider,
        filename: str | Path,
    ):
        super().__init__(config)
        self._app_config = app_config
        self._location_provider = location_provider
        self._filename = Path(filename)
        self._agent = self._construct_agent()
        self._locations: list[LocationData | LocationInfo] | None = None

    @property
    def locations(self) -> list[LocationData | LocationInfo] | None:
        return self._locations

    async def run(self):
        assert self._filename.exists() and self._filename.is_file(), f"File not found: {self._filename}"

        # Load the legend
        legend_path = self._config.get("location_extraction_agent.legend_path", "resources/images/nav_legend.png")
        legend_identifier = self._config.get("location_extraction_agent.legend_identifier", "legend.png")

        if not Path(legend_path).exists():
            raise AssertionError(f"Legend not found: {legend_path}")
        legend = cv2.imread(str(legend_path))
        ok, legend_buf = cv2.imencode(".png", legend)
        assert ok, "Failed to encode legend to PNG"
        legend_data = legend_buf.tobytes()
        assert legend is not None, f"Failed to load legend: {legend_path}"

        img = cv2.imread(str(self._filename))
        assert img is not None, f"Failed to load image: {self._filename}"

        # Ensure coordinates within bounds and ordered
        coords = self._app_config.get("calibration.nav_locations.result_list")
        if coords is None:
            raise AssertionError("Missing calibration.nav_locations.result_list in config")
        x1, y1, x2, y2 = coords.start_x, coords.start_y, coords.end_x, coords.end_y
        h, w = img.shape[:2]
        x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))

        # Crop the image
        cropped = img[y1:y2, x1:x2]

        # Encode the cropped image to PNG
        ok, buf = cv2.imencode(".png", cropped)
        assert ok, "Failed to encode cropped image to PNG"
        data = buf.tobytes()

        # Save crop to screenshots/cropped.png
        if self._config.get("app.screenshots.debug", False):
            save_path = Path(self._config.get("app.screenshots.path", "screenshots"))
            save_path.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path / "cropped.png"), cropped)

        prompt = [
            BinaryContent(
                data=legend_data,
                media_type="image/png",
                identifier=legend_identifier,
            ),
            BinaryContent(
                data=data,
                media_type="image/png",
                identifier=self._filename.name,
            )
        ]

        response = await self._agent.run(prompt)
        output: LocationsResponse = response.output

        logger.debug(f"Extracted locations: {output.locations}")

        output_locations = []
        for location_data in output.locations:
            matched_location = await self._location_provider.identify_location(location_data.name)
            if matched_location is not None:
                output_locations.append(matched_location)
            else:
                logger.warning(f"Failed to identify location: {location_data.name}")
                output_locations.append(location_data)

        self._locations = output_locations

    def _construct_agent(self) -> Agent[None, LocationsResponse]:
        api_key = self._config.get("location_extraction_agent.api_key", "")
        base_url = self._config.get("location_extraction_agent.base_url", None)
        provider = OpenAIProvider(api_key=api_key, base_url=base_url)

        model_id = self._config.get("location_extraction_agent.model_id")
        system_prompt = self._config.get("location_extraction_agent.system_prompt", "")
        instructions = self._config.get("location_extraction_agent.instructions", "")

        model = OpenAIModel(
            provider=provider,
            model_name=model_id,
        )

        agent = Agent[None, LocationsResponse](
            model=model,
            system_prompt=system_prompt,
            output_type=LocationsResponse,
            output_retries=3,
            instructions=instructions,
        )
        return agent