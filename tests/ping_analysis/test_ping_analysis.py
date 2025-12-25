from itertools import batched
import logging
import pytest

from hostile_copilot.config.config import load_config, OmegaConfig
from hostile_copilot.ping_analyzer import (
    PingAnalyzer,
    PingDetection,
    PingAnalysis,
    PingUnknown,
    PingAnalysisResult,
)
from hostile_copilot.ping_analyzer.common import PingAnalysisBase

logger = logging.getLogger(__name__)

#####################################################################
# Helper factory functions
#####################################################################

def PU(rs_value: int) -> PingUnknown:
    return PingUnknown(rs_value=rs_value)

def PAR(rs_value: int, *args) -> PingAnalysisResult:
    detections = []

    for count, label in batched(args, 2):
        detections.append(PingDetection(count=count, label=label))

    return PingAnalysisResult(rs_value=rs_value, prediction=detections)


def analyze_case_id(param):
    """ Used to give a readable name to each test case """
    if isinstance(param, PingUnknown):
        return f"Unknown"

    elif isinstance(param, PingAnalysisResult):
        detection_count = len(param.prediction)
        if detection_count == 1:
            detection = param.prediction[0]
            return f"Result({detection.count}, {detection.label})"

        return f"Result({detection_count} predictions)"
    
    return repr(param)

def build_resource_multiple(config, start=2, end=16):
    """Build test parameters for resource multiples"""
    params = []
    for r in config.ping_analyzer.resources:
        for count in range(start, end + 1):
            params.append((r.rs_value * count, PAR(r.rs_value * count, count, r.label)))
    return sorted(params, key=lambda x: x[0])

def pytest_generate_tests(metafunc):
    mark = metafunc.definition.get_closest_marker("dataset")
    if mark is None:
        return
    
    dataset = mark.args[0]
    config = load_config()
    
    if dataset == "resources/single" and {"rs_value","expected"} <= set(metafunc.fixturenames):
        params = [(r.rs_value, PAR(r.rs_value, 1, r.label)) for r in config.ping_analyzer.resources]
        metafunc.parametrize(("rs_value", "expected"), params, ids=analyze_case_id)

    elif dataset == "resources/multiple" and {"rs_value","expected"} <= set(metafunc.fixturenames):
        params = build_resource_multiple(config)
        metafunc.parametrize(("rs_value", "expected"), params, ids=analyze_case_id)

    elif dataset == "resources/unknowns" and {"rs_value","expected"} <= set(metafunc.fixturenames):
        unknowns = [123, 2001, 1234, 560]
        params = [(i, PingUnknown(rs_value=i)) for i in unknowns if i not in {r.rs_value for r in config.ping_analyzer.resources}]
        metafunc.parametrize(("rs_value", "expected"), params, ids=analyze_case_id)

#####################################################################
# Tests
#####################################################################
class TestPingAnalyzer:
    @pytest.fixture
    def config(self) -> OmegaConfig:
        return load_config()

    @pytest.fixture
    def analyzer(self, config: OmegaConfig) -> PingAnalyzer:
        return PingAnalyzer(config)

    def _test_analyze(self, analyzer: PingAnalyzer, rs_value: int, expected: PingAnalysis):
        result = analyzer.analyze(rs_value)

        assert result == expected
    
    @pytest.mark.dataset("resources/unknowns")
    def test_analyze_unknowns(self, analyzer: PingAnalyzer, rs_value: int, expected: PingAnalysis):
        self._test_analyze(analyzer, rs_value, expected)
    
    @pytest.mark.dataset("resources/single")
    def test_analyze_single(self, analyzer: PingAnalyzer, rs_value: int, expected: PingAnalysis):
        self._test_analyze(analyzer, rs_value, expected)

    @pytest.mark.dataset("resources/multiple")
    def test_analyze_multiple(self, analyzer: PingAnalyzer, rs_value: int, expected: PingAnalysis):
        self._test_analyze(analyzer, rs_value, expected)

    def test_analyze_performance(self, analyzer: PingAnalyzer, config: OmegaConfig):
        import time
        limit = 10000
        
        params = build_resource_multiple(config)
        
        start_time = time.time()
        iterations = 100
        for _ in range(iterations):
            for value, _ in params:
                analyzer.analyze(value)
        end_time = time.time()
        
        elapsed = end_time - start_time

        average_ns = round(elapsed / (iterations * len(params)) * 1e9)
        from pympler import asizeof
        logger.info(f"Average analysis time: {average_ns:,} ns (limit: {limit:,} ns) [{asizeof.asizeof(analyzer._lookup_table)}]")
        assert average_ns < limit, f"Average analysis time {average_ns:,} ns exceeds {limit:,} ns limit"