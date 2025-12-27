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
        self.prediction: list[PingDetection] = sorted(
            predictions,
            key=lambda d: (-d.count, d.label)
        )

    def total_size(self) -> int:
        return sum(d.count for d in self.prediction)

    def __len__(self):
        return len(self.prediction)

    def __iter__(self):
        return iter(self.prediction)

    def __getitem__(self, index):
        return self.prediction[index]

    def __str__(self):
        if len(self.prediction) == 1:
            return self.prediction[0].label
        return ", ".join(f"{d.count}x {d.label}" for d in self.prediction)

    def __repr__(self):
        return f"PingPrediction({self.prediction})"

    def __eq__(self, other):
        if not isinstance(other, PingPrediction):
            return False
        return self.prediction == other.prediction


@dataclass
class PingAnalysisResult(PingAnalysisBase):
    prediction: PingPrediction
    alternates: list[PingPrediction] | None = None

    def __post_init__(self):
        # Ensure prediction is a PingPrediction object
        if not isinstance(self.prediction, PingPrediction):
            self.prediction = PingPrediction(self.prediction)

    def __repr__(self):
        repr = f"Ping [{self.rs_value}] {str(self.prediction)}"

        if self.alternates:
            repr += " | Alternates: " + " | ".join(str(alternate) for alternate in self.alternates)
        return repr

@dataclass
class PingUnknown(PingAnalysisBase):
    def __repr__(self):
        return f"Unknown [{self.rs_value}]"

PingAnalysis = Union[PingAnalysisResult, PingUnknown]