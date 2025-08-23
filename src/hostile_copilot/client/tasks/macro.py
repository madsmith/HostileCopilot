import asyncio
import logging
import pyautogui
from typing import Any

from hostile_copilot.client.tasks.base import Task
from hostile_copilot.config import OmegaConfig

logger = logging.getLogger(__name__)

MacroStep2T = tuple[str, Any | None]
MacroStep3T = tuple[str, Any | None, Any | None]
MacroStepT = MacroStep2T | MacroStep3T

class MacroTask(Task):
    def __init__(self, config: OmegaConfig):
        super().__init__(config)
        self._macro: list[MacroStepT] | None = None
    
    def set_macro(self, macro: list[MacroStepT]):
        self._macro = macro
    
    async def run(self):
        await asyncio.to_thread(self._macro_routine)
    
    def _macro_routine(self):
        if self._macro is None:
            logger.warning("Macro is not set")
            return
            
        for step in self._macro:
            if not isinstance(step, tuple):
                logger.warning(f"Invalid macro step: {step}")
                continue

            logger.debug(f"Executing macro step: {step}")
            if len(step) == 2:
                action, args = step
                kwargs = {}
            elif len(step) == 3:
                action, args, kwargs = step
            else:
                logger.warning(f"Invalid macro step: {step}")
                continue

            if kwargs is None:
                kwargs = {}
            
            if action in pyautogui.__dict__:
                getattr(pyautogui, action)(args, **kwargs)
            else:
                logger.warning(f"Invalid macro action: {action}")