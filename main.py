"""
main.py
-------
CLI entry point for the AI-Based Employee Wellness Management Platform.

Usage:
    python main.py manual "I really enjoy my team."
    python main.py txt data/sample_feedback.txt
    python main.py csv data/sample_feedback.csv

Each run writes a CSV report (and an HTML report) to output/reports/.
"""

import argparse
import logging
import sys
from pathlib import Path

from src import pipeline
from src.emotion import ModelUnavailableError, load_local_model
from src.evaluation import evaluate_saved_models
from src.isear import save_isear_report, validate_isear
from src.preprocessing import ensure_nltk_resources
from src.training import fine_tune
from src.validation import ValidationError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

OUTPUT_CSV = "output/reports/sentiment_report.csv"
OUTPUT_HTML = "output/reports/sentiment_report.html"


def add_emotion_options(parser):
    parser.add_argument("--emotion-model", choices=("bert", "distilbert"),
                        help="Run a previously fine-tuned local emotion model too.")
    parser.add_argument("--model-root", default="models", help="Folder containing saved model folders.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Emotion threshold from 0 to 1.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI-Based Employee Wellness Management Platform — Milestones 1 and 2"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    manual_parser = subparsers.add_parser("manual", help="Ingest a single piece of manual text.")
    manual_parser.add_argument("text", help="The feedback text to analyze.")
    add_emotion_options(manual_parser)

    txt_parser = subparsers.add_parser("txt", help="Ingest feedback from a .txt file.")
    txt_parser.add_argument("path", help="Path to the .txt file.")
    add_emotion_options(txt_parser)

    csv_parser = subparsers.add_parser("csv", help="Ingest feedback from a .csv file.")
    csv_parser.add_argument("path", help="Path to the .csv file.")
    add_emotion_options(csv_parser)

    train_parser = subparsers.add_parser("train", help="Fine-tune and save a BERT or DistilBERT emotion model.")
    train_parser.add_argument("model", choices=("bert", "distilbert"))
    train_parser.add_argument("training_csv", help="CSV with text and labels or six 0/1 emotion columns.")
    train_parser.add_argument("--output-dir", default="models")
    train_parser.add_argument("--epochs", type=int, default=2)
    train_parser.add_argument("--batch-size", type=int, default=4)

    evaluate_parser = subparsers.add_parser("evaluate", help="Compare both saved models on a held-out labelled CSV.")
    evaluate_parser.add_argument("heldout_csv")
    evaluate_parser.add_argument("--model-root", default="models")
    evaluate_parser.add_argument("--threshold", type=float, default=0.5)

    isear_parser = subparsers.add_parser("isear", help="Validate one saved model on a separate local ISEAR subset.")
    isear_parser.add_argument("model", choices=("bert", "distilbert"))
    isear_parser.add_argument("csv_path")
    isear_parser.add_argument("--model-root", default="models")
    isear_parser.add_argument("--threshold", type=float, default=0.5)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.mode == "train":
            saved = fine_tune(args.model, args.training_csv, args.output_dir, args.epochs, args.batch_size)
            print(f"Saved {args.model} model and tokenizer: {saved}")
            return 0
        if args.mode == "evaluate":
            evaluated = evaluate_saved_models(args.heldout_csv, args.model_root, args.threshold)
            print("Model        Accuracy  Precision  Recall  Macro F1")
            for row in evaluated["comparison"]:
                print(f"{row['model']:<12} {row['accuracy']:.4f}    {row['precision']:.4f}     {row['recall']:.4f}  {row['macro_f1']:.4f}")
            print(f"Selected model: {evaluated['selection']['best_model']} ({evaluated['selection']['selection_rule']})")
            return 0
        if args.mode == "isear":
            model, tokenizer = load_local_model(Path(args.model_root) / args.model)
            benchmark = validate_isear(model, tokenizer, args.csv_path, args.threshold)
            paths = save_isear_report(benchmark, "output/reports")
            print(f"ISEAR samples: {benchmark['sample_count']}; subset accuracy: {benchmark['metrics']['accuracy']:.4f}")
            print(f"ISEAR reports: {paths['json_path']} and {paths['html_path']}")
            return 0
        ensure_nltk_resources()
        if args.mode in {"manual", "txt", "csv"} and args.emotion_model:
            model, tokenizer = load_local_model(Path(args.model_root) / args.emotion_model)
            runners = {"manual": pipeline.run_manual_with_emotion, "txt": pipeline.run_txt_with_emotion,
                       "csv": pipeline.run_csv_with_emotion}
            value = args.text if args.mode == "manual" else args.path
            result = runners[args.mode](value, OUTPUT_CSV, OUTPUT_HTML, model, tokenizer, args.threshold)
        elif args.mode == "manual":
            result = pipeline.run_manual(args.text, OUTPUT_CSV, OUTPUT_HTML)
        elif args.mode == "txt":
            result = pipeline.run_txt(args.path, OUTPUT_CSV, OUTPUT_HTML)
        elif args.mode == "csv":
            result = pipeline.run_csv(args.path, OUTPUT_CSV, OUTPUT_HTML)
        else:  # pragma: no cover - argparse guards this
            raise ValueError(f"Unknown mode: {args.mode}")
    except (ValidationError, ValueError, ModelUnavailableError, RuntimeError) as exc:
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
