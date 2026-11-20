"""Command-line entry point: `python -m app.cli --help`."""
import argparse

from app.core import milestone_1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="legal-document-lora-tagger",
        description="Public sector clerks classify documents faster. This tool uses LoRA to fine-tune GPT-2 to tag legal clauses in public records, improving efficiency and reducing manual review time.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")

    prep = sub.add_parser("prepare", help="Milestone 1: prepare legal clause dataset")
    prep.add_argument("--samples-per-class", type=int, default=120)

    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Legal Document LoRA Tagger is ready. Open README.md for your milestones.")
    elif args.command == "prepare":
        stats = milestone_1(samples_per_class=args.samples_per_class)
        print(f"Prepared dataset: {stats}")


if __name__ == "__main__":
    main()