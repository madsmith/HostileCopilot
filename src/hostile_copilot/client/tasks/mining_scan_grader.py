import logging

from hostile_copilot.client.voice import VoiceClient
from hostile_copilot.config import OmegaConfig
from hostile_copilot.client.commodity_grader import CommodityGrader, ScanGrade, CommodityGrade
from hostile_copilot.utils.speech import roundify_numbers

from .base import Task
from .types import ScanResponse

logger = logging.getLogger(__name__)

class MiningScanGraderTask(Task):
    def __init__(
        self,
        config: OmegaConfig,
        app_config: OmegaConfig,
        scan_result: ScanResponse,
        voice_client: VoiceClient,
        commodity_grader: CommodityGrader,
        tool_mode: bool = False
    ):
        super().__init__(config)
        self._app_config = app_config
        self._scan_result = scan_result
        self._voice_client = voice_client
        self._commodity_grader = commodity_grader
        self._tool_mode = tool_mode    
    
    async def run(self):
        scan_data = self._scan_result.scan_data
        if scan_data is None:
            if self._tool_mode:
                return "No scan data found"
            else:
                await self._voice_client.speak("No scan data found")
            return

        scan_grade: ScanGrade = self._commodity_grader.grade_scan(scan_data)

        logger.debug(f"Scan grade: {scan_grade}")

        notify_value = self._config.get("app.scanning.notify_levels.value", 500000)
        notify_tier = self._config.get("app.scanning.notify_levels.tier", 3)
        notify_single_item_value = self._config.get("app.scanning.notify_levels.single_item_value", 300000)

        msg = None
        # Get best commodity from scan data
        best_commodity: CommodityGrade | None = next(
            (commodity for commodity in scan_grade.commodity_grades if commodity.tier == scan_grade.best_tier),
            None
        )
        
        if best_commodity is None:
            logger.warning("No best commodity found in scan data")
            if self._tool_mode:
                return "Scan data error"
            else:
                await self._voice_client.speak("Scan data error")
            return
        
        if scan_grade.total_value >= notify_value:
            msg = f"High Value. {scan_grade.total_value} A.U.E.C. with {best_commodity.value} of {best_commodity.commodity}"
        elif scan_grade.best_tier <= notify_tier:
            msg = f"High Tier. {best_commodity.size} SCU of {best_commodity.commodity}, valued at {best_commodity.value} A.U.E.C."
        else:
            for commodity in scan_grade.commodity_grades:
                if commodity.value >= notify_single_item_value:
                    msg = f"High Value Single Item. {commodity.size} SCU of {commodity.commodity}, valued at {commodity.value} A.U.E.C."
                    break
        if msg:
            msg = roundify_numbers(msg)
            if self._tool_mode:
                return msg
            else:
                await self._voice_client.speak(msg)
            return




        