import gc, logging, time, threading, collections, asyncio, concurrent.futures, numpy as np
logger = logging.getLogger(__name__)

WATCH_TYPES = (bytes, bytearray, np.ndarray, asyncio.Task, concurrent.futures.Future)

def dump_counts_every(seconds=60):
    def _run():
        while True:
            time.sleep(seconds)
            objs = gc.get_objects()
            ctr = collections.Counter(type(o) for o in objs if isinstance(o, WATCH_TYPES))
            logger.info("=== Type counts ===")
            for t, c in ctr.most_common():
                logger.info("%s: %d", getattr(t, '__name__', str(t)), c)
    threading.Thread(target=_run, daemon=True).start()