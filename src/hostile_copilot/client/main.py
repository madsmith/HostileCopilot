#!/usr/bin/env python3
"""
Main entry point for HostileCoPilot

This module provides the command-line interface for the HostileCoPilot application.
"""

import asyncio
import argparse
import logging



async def run_client() -> None:
    """Run the HostileCoPilot client"""
    logging.info("Starting HostileCoPilot client...")

    is_running = True
    while is_running:
        try:
            logging.debug("idle...")
            await asyncio.sleep(1)
        except KeyboardInterrupt:
            is_running = False


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
    
    asyncio.run(run_client())


if __name__ == "__main__":
    main()
