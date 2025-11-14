import asyncio
import logging
import pyautogui
from typing import Any

from hostile_copilot.client.tasks.base import Task
from hostile_copilot.config import OmegaConfig
from hostile_copilot.utils.input.keyboard import Keyboard

logger = logging.getLogger(__name__)

MacroStep2T = tuple[str, Any | None]
MacroStep3T = tuple[str, Any | None, Any | None]
MacroStepT = MacroStep2T | MacroStep3T



keyboard_actions = {
    "vkbd:press": Keyboard.press_key,
    "vkbd:sequence": Keyboard.type_sequence,
    "vkbd:sleep": Keyboard.asyncSleep,
}

class MacroTask(Task):
    def __init__(self, config: OmegaConfig, keyboard: Keyboard):
        super().__init__(config)
        self._macro: list[MacroStepT] | None = None
        self._keyboard = keyboard
    
    def set_macro(self, macro: list[MacroStepT]):
       self._macro = macro
    
    async def run(self):
        await self._macro_routine()
    
    async def _macro_routine(self):
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

            if isinstance(args, tuple):
                args = list(args)
            if not isinstance(args, list):
                args = [args]

            if kwargs is None:
                kwargs = {}

            if action in keyboard_actions:
                handler = keyboard_actions[action]
                logger.debug(f"Executing keyboard action: {action}")
                await handler(self._keyboard, *args, **kwargs)
            elif action in pyautogui.__dict__:
                logger.debug(f"Executing pyautogui action: {action} with args: {args} and kwargs: {kwargs}")
                getattr(pyautogui, action)(*args, **kwargs)
            else:
                logger.warning(f"Invalid macro action: {action}")