import argparse
import logging
import sys

from lm_agent.core import agentic_run, check_server
from lm_agent.router import route_and_respond

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    parser = argparse.ArgumentParser(description="LM Studio CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--agent", type=str, help="Run the full agentic loop across tools")
    group.add_argument("--route", type=str, help="Route to best model without tools")
    
    args = parser.parse_args()

    if not check_server():
        sys.exit(1)

    if args.agent:
        print(f"Agent Request: {args.agent}")
        print("-" * 40)
        res = agentic_run(args.agent)
        print(f"\n{res}")
        
    elif args.route:
        print(f"Route Request: {args.route}")
        print("-" * 40)
        res = route_and_respond(args.route)
        print(f"\n{res}")

if __name__ == "__main__":
    main()
