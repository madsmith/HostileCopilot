import asyncio
import argparse
import io
from pathlib import Path
from PIL import Image
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent, BinaryContent

from hostile_copilot.config import load_config

from hostile_copilot.client.tasks import PingResponse

async def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, default="./config/config.yaml", help="Path to configuration file")
    argparser.add_argument("--app-config", type=str, default="./config/settings.yaml", help="Path to application configuration file")
    argparser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    argparser.add_argument("screenshot", type=str, help="Path to screenshot file")
    args = argparser.parse_args()

    config = load_config(args.config)
    app_config = load_config(args.app_config)

    api_key = config.get("ping_analysis_agent.api_key", "")
    base_url = config.get("ping_analysis_agent.base_url", None)
    provider = OpenAIProvider(api_key=api_key, base_url=base_url)
    
    model_id = config.get("ping_analysis_agent.model_id")
    system_prompt = config.get("ping_analysis_agent.system_prompt", "")
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

    filename = args.screenshot

    image_path = Path(filename)

    if not image_path.exists():
        print(f"File not found: {filename}")
        return

    image = Image.open(image_path)

    coordinates = app_config.get("calibration.ping_scan")

    if not coordinates:
        print("Ping scan coordinates not found in app config")
        return

    bounding_box = (
        coordinates.start_x,
        coordinates.start_y,
        coordinates.end_x,
        coordinates.end_y
    )

    cropped_image = image.crop(bounding_box)

    if args.debug:
        cropped_image.save("screenshots/screenshot.png")

    buffer = io.BytesIO()
    cropped_image.save(buffer, format="PNG")
    binary_image = buffer.getvalue()

    prompt = [
        BinaryContent(
            data=binary_image,
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