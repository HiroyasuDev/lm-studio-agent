import argparse
import asyncio
import logging
import sys

from lm_agent.core import agentic_run, check_server
from lm_agent.router import route_and_respond

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

async def async_main():
    parser = argparse.ArgumentParser(description="LM Studio Async Orchestrator CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--agent", type=str, help="Launch complete asynchronous agent pipeline")
    group.add_argument("--route", type=str, help="Asynchronously route prompt to optimal model bounds")
    
    args = parser.parse_args()

    # Pre-flight health check
    server_healthy = await check_server()
    if not server_healthy:
        sys.exit(1)

    if args.agent:
        print(f"\n[ASYNC ORCHESTRATION INITIATED]: {args.agent}\n" + "-" * 50)
        res = await agentic_run(args.agent)
        print(f"\n[ORCHESTRATION COMPLETE]\n{res}\n")
        
    elif args.route:
        print(f"\n[ROUTING INITIATED]: {args.route}\n" + "-" * 50)
        res = await route_and_respond(args.route)
        print(f"\n[ROUTING COMPLETE]\n{res}\n")

def main():
    """Synchronous entry point bootloader for the asyncio event loop."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n[ABORT] Orchestration actively terminated by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
