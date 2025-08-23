#!/usr/bin/env python3
"""
Main entry point for HostileCoPilot

This module provides the command-line interface for the HostileCoPilot application.
"""

import asyncio
import argparse
import logging

from hostile_copilot.client.app import HostileCoPilotApp
from hostile_copilot.config import load_config
from hostile_copilot.utils.debug.tracemalloc import start_tracemalloc, dump_tracemalloc_diff_every
from hostile_copilot.utils.debug.objects import dump_object_summary_every
from hostile_copilot.utils.debug.counts import dump_counts_every
from hostile_copilot.utils.debug.objects import dump_gc_objects_every

async def run_app(args: argparse.Namespace) -> None:
    # start_tracemalloc()
    # dump_tracemalloc_diff_every(
    #     seconds=30,
    #     top=20,
    #     include_patterns=["*hostile_copilot*"],  # focus on repo
    #     key="traceback",
    # )
    # dump_object_summary_every(30)
    # dump_counts_every(30)
    # dump_gc_objects_every(120)
    
    """Run the HostileCoPilot client"""
    logging.info("Starting HostileCoPilot client...")

    config = load_config()
    
    app = HostileCoPilotApp(config)
    await app.initialize()
    try:
        await app.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass

def main() -> None:
    """Main entry point for the application"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version="HostileCoPilot 0.1.0")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Setup logging format
    logging.basicConfig(
        format='%(levelname)-8s | %(name)s | %(message)s',
        level=logging.INFO
    )
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        suppress_warnings()
    
    asyncio.run(run_app(args))

def suppress_warnings():
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    suppress_warnings()
    main()
