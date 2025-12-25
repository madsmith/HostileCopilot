from dataclasses import dataclass

from hostile_copilot.config import OmegaConfig

from .common import (
    PingAnalysis,
    PingUnknown,
    PingAnalysisResult,
    PingDetection,
    PingPrediction,
)

@dataclass
class LookupEntry:
    prediction: PingPrediction
    alternates: list[PingPrediction] | None = None

class PingAnalyzer:

    def __init__(self, config: OmegaConfig):
        self._config = config
        self._resource_data = config.ping_analyzer.resources
        self._lookup_table: dict[int, LookupEntry] = {}

        self._initialize_lookup_tables(18)

    def _initialize_lookup_tables(self, max_multiple: int) -> None:
        for i in range(1, max_multiple + 1):
            for resource in self._resource_data:
                value = i * resource.rs_value
                prediction: PingPrediction = [PingDetection(count=i, label=resource.label)]

                if value in self._lookup_table:
                    if self._lookup_table[value].alternates is None:
                        self._lookup_table[value].alternates = [prediction]
                    else:
                        self._lookup_table[value].alternates.append(prediction)
                else:
                    self._lookup_table[value] = LookupEntry(prediction=prediction, alternates=None)


    def analyze(self, rs_value: int) -> PingAnalysis:
        # 1. Attempt lookup
        if rs_value in self._lookup_table:
            entry = self._lookup_table[rs_value]
            return PingAnalysisResult(
                rs_value=rs_value, 
                prediction=entry.prediction,
                alternates=entry.alternates if entry.alternates else None
            )

        # 2. Calculate detection
        detections = self._process_detection(rs_value)

        if detections is None:
            return PingUnknown(rs_value=rs_value)

        return PingAnalysisResult(rs_value=rs_value, prediction=detections)


    def _process_detection(self, rs_value: int) -> list[PingDetection] | None:
        candidate = None

        for resource in self._resource_data:
            count = rs_value // resource.rs_value
            remainder = rs_value % resource.rs_value

            if remainder == 0:
                detection = [PingDetection(count=count, label=resource.label)]
                if candidate is None:
                    candidate = detection
                elif detection[0].count < candidate[0].count:
                    candidate = detection
        
        return candidate

        
