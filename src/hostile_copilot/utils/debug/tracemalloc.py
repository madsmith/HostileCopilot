# utils/debug/tracemalloc.py
import tracemalloc, time, logging, os, threading
logger = logging.getLogger(__name__)

def start_tracemalloc(frames: int = 25):
    if not tracemalloc.is_tracing():
        tracemalloc.start(frames)

def dump_tracemalloc_diff_every(
    seconds: int = 60,
    top: int = 15,
    include_patterns: list[str] | None = None,
    key: str = "traceback",  # 'lineno' | 'filename' | 'traceback'
):
    # Filter only our code to cut noise
    filters = []
    if include_patterns:
        for pat in include_patterns:
            filters.append(tracemalloc.Filter(True, pat))
    else:
        # default: your repo path
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        filters.append(tracemalloc.Filter(True, f"{root}*"))

    def _run():
        prev = tracemalloc.take_snapshot().filter_traces(filters)
        while True:
            time.sleep(seconds)
            snap = tracemalloc.take_snapshot().filter_traces(filters)
            stats = snap.compare_to(prev, key_type=key)
            logger.info("=== Tracemalloc diff (top %d by %s) ===", top, key)
            for s in stats[:top]:
                logger.info("%s", s)
                # Print a few frames of the traceback for the stat
                for frame in s.traceback.format()[-6:]:
                    logger.info("  %s", frame)
            prev = snap

    threading.Thread(target=_run, daemon=True).start()