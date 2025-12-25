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
        self._resource_data = sorted(
            config.ping_analyzer.resources,
            key=lambda r: r.rs_value,
            reverse=True
        )
        self._lookup_table: dict[int, LookupEntry] = {}
        self._label_to_location_group = {
            r.label: r.location_group for r in config.ping_analyzer.resources
        }

        self._initialize_lookup_tables(18)

    def _initialize_lookup_tables(self, max_multiple: int) -> None:
        for i in range(1, max_multiple + 1):
            for resource in self._resource_data:
                value = i * resource.rs_value
                prediction: PingPrediction = PingPrediction([PingDetection(count=i, label=resource.label)])

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
        candidate: PingPrediction | None = None

        for resource in self._resource_data:
            count = rs_value // resource.rs_value
            remainder = rs_value % resource.rs_value
            group = self._label_to_location_group.get(resource.label, "unknown")

            if remainder == 0:
                detection = PingPrediction([PingDetection(count=count, label=resource.label)])
                if candidate is None:
                    candidate = detection
                elif detection.total_size() < candidate.total_size():
                    candidate = detection
            else:
                # Search if the remainder can be found in another component.
                for subtract_count in range(0, 2):
                    check_remainder = remainder + subtract_count * resource.rs_value
                    check_count = count - subtract_count

                    if check_count <= 0:
                        continue

                    if check_remainder in self._lookup_table:
                        remainder_prediction = self._lookup_table[check_remainder].prediction

                        # Remainder can not be the same resource type
                        if remainder_prediction.prediction[0].label == resource.label:
                            continue

                        # Remainder must match location group
                        remainder_group = self._label_to_location_group.get(
                            remainder_prediction.prediction[0].label,
                            "unknown"
                        )
                        if remainder_group != group:
                            continue

                        # Found a valid combination
                        main_detection: PingDetection = PingDetection(count=check_count, label=resource.label)
                        combined = PingPrediction([main_detection] + remainder_prediction.prediction)
                        if candidate is None or combined.total_size() < candidate.total_size():
                            candidate = combined
                        break

                # if remainder in self._lookup_table and candidate is None:
                #     # Combine the main resource with the remainder
                #     remainder_prediction = self._lookup_table[remainder].prediction
                #     remainder_group = self._label_to_location_group.get(
                #         remainder_prediction.predictions[0].label,
                #         "unknown"
                #     )
                #     if remainder_group != group:
                #         print(f"Skipping remainder {remainder_prediction} with {resource}")
                #         continue
                #     main_detection: PingDetection = PingDetection(count=count, label=resource.label)
                #     combined = PingPrediction([main_detection] + remainder_prediction.predictions)
                #     if candidate is None or combined.total_size() < candidate.total_size():
                #         candidate = combined
        
        return candidate

        
