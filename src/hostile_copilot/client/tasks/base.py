from abc import ABC, abstractmethod

from hostile_copilot.config import OmegaConfig

class Task(ABC):
    def __init__(self, config: OmegaConfig):
        self._config = config

    @abstractmethod
    async def run(self):
        pass