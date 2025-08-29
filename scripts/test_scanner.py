
import asyncio
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from typing import Literal, Optional
import logging
import logfire


logfire_instance = logfire.configure(service_name="test_scanner", environment="development", console=False)

logfire_instance.instrument_pydantic_ai()
logfire_instance.instrument_requests()

logging.basicConfig(level=logging.DEBUG)


from hostile_copilot.config import load_config



class ScanItem(BaseModel):
    material: str = Field(
        description="The name of the material"
    )
    percentage: float = Field(
        description="The percentage of the material"
    )

    @field_validator("percentage", mode="before")
    @classmethod
    def normalize_percentage(cls, v):
        if isinstance(v, str):
            v = v.strip().replace("%", "")
        try:
            return float(v)
        except ValueError:
            raise ValueError(f"Invalid percentage: {v}")

DifficultyT = Literal["Easy", "Medium", "Hard", "Impossible", "Unknown"]

class ScanData(BaseModel):
    object_type: str = Field(
        description="The type of object scanned. e.g. Asteroid, Sal"
    )
    object_subtype: str | None = Field(
        description="The subtype of the object scanned.  Usually found within parentheses.. e.g. Q-Type"
    )
    mass: int = Field(
        description="The mass of the object"
    )
    resistance: int = Field(
        description="The resistance of the object as a percentage"
    )
    instability: float = Field(
        description="The instability of the object"
    )
    difficulty: Optional[DifficultyT] = Field(
        description="The difficulty rating from Easy to Impossible.  Not always present in scan."
    )
    size: float = Field(
        description="The size of the object measured in SCU"
    )
    composition: list[ScanItem] = Field(
        description="Mapping of the material name to percentage composition found in the scan results",
        min_items=1
    )
    # composition: Composition = Field(
    #     description="The composition of the object"
    # )

    @field_validator("difficulty", mode="before")
    @classmethod
    def normalize_difficulty(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if isinstance(v, str):
            val = v.strip().lower()
            if val in {"unknown", "unk", "n/a", "none"}:
                return "Unknown"
            if val == "easy":
                return "Easy"
            if val == "medium":
                return "Medium"
            if val == "hard":
                return "Hard"
            if val == "impossible":
                return "Impossible"
        return v

class ScanResponse(BaseModel):
    scan_data: Optional[ScanData] = Field(
        None,
        description="The parsed scan.  If no scan data is present in the image, return null."
    )
    

async def main():
    config = load_config()

    api_key = config.get("mining_scan_agent.api_key", "")
    base_url = config.get("mining_scan_agentagent.base_url", None)
    provider = OpenAIProvider(api_key=api_key, base_url=base_url)

    model_id = config.get("mining_scan_agent.model_id")
    system_prompt = config.get("mining_scan_agent.system_prompt", "")
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

    filename = "./cropped_image.png"
    with open(filename, "rb") as f:
        image = f.read()

    prompt = [
        BinaryContent(
            data=image,
            media_type="image/png",
            identifier="filename"
        )
    ]
    response = await agent.run(prompt)
    data = response.output
    print(type(data))
    print(data.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(main())