from pydantic_ai import Agent, BinaryContent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel

from hostile_copilot.config import OmegaConfig

from .base import Task
from .bounded_screenshot import BoundedScreenshotTask
from .types import ScanResponse


class MiningScanTask(Task):
    def __init__(self, config: OmegaConfig, app_config: OmegaConfig):
        super().__init__(config)
        self._app_config = app_config

        self._agent = self._construct_agent()

        self._scan_result: ScanResponse | None = None

    @property
    def scan_result(self) -> ScanResponse | None:
        return self._scan_result

    async def run(self):
        self._scan_result = None

        coordinates = self._app_config.get("calibration.mining_scan")

        bounding_box = (
            coordinates.start_x,
            coordinates.start_y,
            coordinates.end_x,
            coordinates.end_y
        )

        task = BoundedScreenshotTask(self._config, bounding_box)
        await task.run()

        image_data = task.binary_encoded()

        prompt = [
            BinaryContent(
                data=image_data,
                media_type="image/png",
                identifier="filename"
            )
        ]

        response = await self._agent.run(prompt)
        data = response.output

        self._scan_result = data

    def _construct_agent(self) -> Agent[None, ScanResponse]:
        api_key = self._config.get("mining_scan_agent.api_key", "")
        base_url = self._config.get("mining_scan_agentagent.base_url", None)
        provider = OpenAIProvider(api_key=api_key, base_url=base_url)

        model_id = self._config.get("mining_scan_agent.model_id")
        system_prompt = self._config.get("mining_scan_agent.system_prompt", "")
        model = OpenAIModel(
            provider=provider,
            model_name=model_id,
        )


        agent = Agent[None, ScanResponse](
            model=model,
            system_prompt=system_prompt,
            output_type=ScanResponse,
            output_retries=3,
            instructions="Extract the scan data from the image"
        )

        return agent