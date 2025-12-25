from dataclasses import dataclass
from typing import Union

@dataclass(frozen=True, eq=True)
class PingDetection:
    count: int
    label: str

@dataclass
class PingAnalysisBase:
    rs_value: int


class PingPrediction:
    def __init__(self, predictions: list[PingDetection]):
        # Maintain a stable, deterministic ordering for indexed access
        self.predictions: list[PingDetection] = sorted(
            predictions,
            key=lambda d: (-d.count, d.label)
        )

    def total_size(self) -> int:
        return sum(d.count for d in self.predictions)

    def __len__(self):
        return len(self.predictions)

    def __iter__(self):
        return iter(self.predictions)

    def __getitem__(self, index):
        return self.predictions[index]

    def __repr__(self):
        return f"PingPrediction({self.predictions})"

    def __eq__(self, other):
        if not isinstance(other, PingPrediction):
            return False
        return self.predictions == other.predictions


@dataclass
class PingAnalysisResult(PingAnalysisBase):
    prediction: PingPrediction
    alternates: list[PingPrediction] | None = None

    def __post_init__(self):
        # Ensure prediction is a PingPrediction object
        if not isinstance(self.prediction, PingPrediction):
            self.prediction = PingPrediction(self.prediction)


@dataclass
class PingUnknown(PingAnalysisBase):
    pass

PingAnalysis = Union[PingAnalysisResult, PingUnknown]