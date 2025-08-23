from pympler import muppy, summary
import logging, time, threading
from itertools import islice
import gc
import numpy as np
import asyncio
import concurrent.futures
logger = logging.getLogger(__name__)

def dump_object_summary_every(seconds=120):
    def _run():
        while True:
            try:
                time.sleep(seconds)
                all_objs = muppy.get_objects()
                sum1 = summary.summarize(all_objs)
                logger.info("=== Pympler summary ===")
                for line in islice(summary.format_(sum1), 40):  # top lines only
                    logger.info(line)
            except Exception:
                logger.exception("Error while generating Pympler object summary")
    threading.Thread(target=_run, daemon=True).start()

WATCH_TYPES = (bytes, bytearray, np.ndarray, asyncio.Task, concurrent.futures.Future)

def dump_gc_objects_every(seconds=120):
    def _run():
        while True:
            try:
                time.sleep(seconds)
                all_objs = gc.get_objects()
                logger.info("=== GC objects ===")
                for obj in all_objs:
                    if isinstance(obj, WATCH_TYPES):
                        logger.info(obj)
            except Exception:
                logger.exception("Error while generating GC object summary")
    threading.Thread(target=_run, daemon=True).start()