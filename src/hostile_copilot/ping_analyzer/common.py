from dataclasses import dataclass
from typing import Union

@dataclass
class PingDetection:
    count: int
    label: str

@dataclass
class PingAnalysisBase:
    rs_value: int

PingPrediction = list[PingDetection]

@dataclass
class PingAnalysisResult(PingAnalysisBase):
    prediction: PingPrediction
    alternates: list[PingDetection] | None = None


@dataclass
class PingUnknown(PingAnalysisBase):
    pass

PingAnalysis = Union[PingAnalysisResult, PingUnknown]