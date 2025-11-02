import requests
import json
import logging
import time
from pathlib import Path
from typing import Any

from hostile_copilot.config import OmegaConfig
from .types import GravityWell

logger = logging.getLogger(__name__)


class RegolithException(Exception):
    pass


class RegolithClient:
    def __init__(self, config: OmegaConfig):
        self._config = config
        self._base_url: str = config.get("regolith.api_url", "https://api.regolith.rocks")
        self._api_key: str | None = config.get("regolith.api_key", None)
        self._file_cache_path: Path = Path(config.get("regolith.file_cache_path", "cache/regolith"))
        self._cache_expiry: int = config.get("regolith.cache_expiry", 24 * 3600)

        assert self._api_key is not None, "Missing config 'regolith.api_key'"

    async def graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        url = self._base_url
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
        }

        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables

        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            raise RegolithException(f"Request failed: {response.status_code} - {response.text}")

        try:
            data = response.json()
        except ValueError:
            raise RegolithException("Failed to parse JSON response")

        if "errors" in data and data["errors"]:
            raise RegolithException(f"GraphQL errors: {data['errors']}")

        if "data" not in data:
            raise RegolithException("No 'data' field in GraphQL response")

        return data["data"]

    async def graphql_cached(self, cache_key: str, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        cache_path = self._file_cache_path / f"{cache_key}.json"

        if await self._is_cached(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.exception(f"Failed to read cache '{cache_path}': {e}")
                try:
                    cache_path.unlink(missing_ok=True)
                except Exception:
                    pass

        data = await self.graphql(query, variables)
        await self._cache_data(cache_path, data)
        return data

    async def _cache_data(self, cache_path: Path, data: dict[str, Any]):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")

    async def _is_cached(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        return (time.time() - cache_path.stat().st_mtime) < self._cache_expiry

    #################################################################
    # High-level helpers
    #################################################################

    async def fetch_uex_bodies(self) -> list[GravityWell]:
        query = (
            """
            query getPublicLookups {
              lookups {
                UEX {
                  bodies
                  __typename
                }
                __typename
              }
            }
            """
        )

        data = await self.graphql_cached("lookups_uex_bodies", query)
        # structure: { 'lookups': { 'UEX': { 'bodies': [...] } } }
        try:
            raw_bodies = data["lookups"]["UEX"]["bodies"]
            return [GravityWell(**item) for item in raw_bodies]
        except Exception as e:
            raise RegolithException(f"Unexpected response structure: {e}")
