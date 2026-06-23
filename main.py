import argparse
import logging
import os
import sys

from src.cli import run_cli


def _configure_logging():
    """Send structured logs to qtrade.log so the logger.* calls in the pipeline
    are actually captured. Level is controllable via the LOG_LEVEL env var."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[logging.FileHandler("qtrade.log")],
    )
    # Quiet noisy third-party loggers so the console stays readable.
    for noisy in ("httpx", "LiteLLM", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main():
    """Primary execution router for the QTrade AI Support microservice."""
    _configure_logging()

    parser = argparse.ArgumentParser(description="QTrade AI Support Assistant")
    parser.add_argument(
        "--api", action="store_true", help="Run the HTTP API instead of the CLI"
    )
    parser.add_argument("--host", default="127.0.0.1", help="API host (with --api)")
    parser.add_argument("--port", type=int, default=8000, help="API port (with --api)")
    args = parser.parse_args()

    try:
        if args.api:
            from src.api import run_api

            run_api(host=args.host, port=args.port)
        else:
            run_cli()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()