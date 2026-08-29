"""
main.py
-------
CLI entry point for the AI-Based Employee Wellness Management Platform
(Milestone 1: Text Ingestion & Baseline Sentiment).

Usage:
    python main.py manual "I really enjoy my team."
    python main.py txt data/sample_feedback.txt
    python main.py csv data/sample_feedback.csv

Each run writes a CSV report (and an HTML report) to output/reports/.
"""

import argparse
import logging
import sys

from src import pipeline
from src.preprocessing import ensure_nltk_resources
from src.validation import ValidationError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

OUTPUT_CSV = "output/reports/sentiment_report.csv"
OUTPUT_HTML = "output/reports/sentiment_report.html"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI-Based Employee Wellness Management Platform — Milestone 1 CLI"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    manual_parser = subparsers.add_parser("manual", help="Ingest a single piece of manual text.")
    manual_parser.add_argument("text", help="The feedback text to analyze.")

    txt_parser = subparsers.add_parser("txt", help="Ingest feedback from a .txt file.")
    txt_parser.add_argument("path", help="Path to the .txt file.")

    csv_parser = subparsers.add_parser("csv", help="Ingest feedback from a .csv file.")
    csv_parser.add_argument("path", help="Path to the .csv file.")

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    ensure_nltk_resources()

    try:
        if args.mode == "manual":
            result = pipeline.run_manual(args.text, OUTPUT_CSV, OUTPUT_HTML)
        elif args.mode == "txt":
            result = pipeline.run_txt(args.path, OUTPUT_CSV, OUTPUT_HTML)
        elif args.mode == "csv":
            result = pipeline.run_csv(args.path, OUTPUT_CSV, OUTPUT_HTML)
        else:  # pragma: no cover - argparse guards this
            raise ValueError(f"Unknown mode: {args.mode}")
    except ValidationError as exc:
        logger.error("Input rejected: %s", exc)
        print(f"ERROR: {exc}")
        return 1
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        print(f"ERROR: {exc}")
        return 1

    summary = result["summary"]
    print("\n--- Milestone 1 Run Summary ---")
    print(f"Total Samples : {summary['total_samples']}")
    print(f"Positive      : {summary['positive_count']}")
    print(f"Negative      : {summary['negative_count']}")
    print(f"Neutral       : {summary['neutral_count']}")
    print(f"CSV Report    : {result['csv_path']}")
    if result["html_path"]:
        print(f"HTML Report   : {result['html_path']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
