from dataclasses import dataclass
import logging
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel

from hostile_copilot.config import OmegaConfig

from .base import Task
from .bounded_screenshot import BoundedScreenshotTask
from .types import (
    PingResponse,
    PingData,
    PingResourceSummary,
    PingSignatureReadout
)

logger = logging.getLogger(__name__)

@dataclass
class PingAnalysisResult:
    name: str
    count: int

class PingAnalysisTask(Task):
    def __init__(self, config: OmegaConfig, app_config: OmegaConfig):
        super().__init__(config)
        self._app_config = app_config

        self._agent = self._construct_agent()

        self._ping_result: PingResponse | None = None
    
    @property
    def ping_result(self) -> PingResponse | None:
        return self._ping_result
    
    async def run(self):
        self._ping_result = None

        coordinates = self._app_config.get("calibration.ping_scan")

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

        self._ping_result = data
        pass

    def _construct_agent(self) -> Agent[None, PingResponse]:
        api_key = self._config.get("ping_analysis_agent.api_key", "")
        base_url = self._config.get("ping_analysis_agentagent.base_url", None)
        provider = OpenAIProvider(api_key=api_key, base_url=base_url)

        model_id = self._config.get("ping_analysis_agent.model_id")
        system_prompt = self._config.get("ping_analysis_agent.system_prompt", "")
        model = OpenAIModel(
            provider=provider,
            model_name=model_id,
        )


        agent = Agent[None, PingResponse](
            model=model,
            system_prompt=system_prompt,
            output_type=PingResponse,
            output_retries=3,
            instructions="Extract the ping data from the image"
        )

        return agent

    def process_ping(self, ping_data: PingData) -> PingAnalysisResult | PingSignatureReadout | None:
        if isinstance(ping_data, PingResourceSummary):
            return self.analyze_ping_resource_summary(ping_data)
        elif isinstance(ping_data, PingSignatureReadout):
            return ping_data
        else:
            return None

    def analyze_ping_resource_summary(self, ping_data: PingResourceSummary) -> PingAnalysisResult | None:
        assert isinstance(ping_data, PingResourceSummary), f"Expected PingResourceSummary not {type(ping_data)}"
        ping_mapping = self._config.get("ping_analysis_agent.resource_mapping", None)

        if ping_mapping is None:
            logger.warning("Missing ping mapping in config")
            return None

        match: PingAnalysisResult | None = None
        for radar_value, item_name in ping_mapping.items():
            if ping_data.radar % radar_value == 0:
                count = ping_data.radar // radar_value

                if match is None or count < match.count:
                    match = PingAnalysisResult(item_name, count)

        if match is None:
            match = PingAnalysisResult("Unknown Radar Signature", 1)
        
        return match
        