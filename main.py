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
    try:
        run_cli()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()