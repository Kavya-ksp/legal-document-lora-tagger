"""Command-line entry point: `python -m app.cli --help`."""
import argparse

from app.core import milestone_1, milestone_2, milestone_3


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="legal-document-lora-tagger",
        description="Public sector clerks classify documents faster. This tool uses LoRA to fine-tune GPT-2 to tag legal clauses in public records, improving efficiency and reducing manual review time.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")

    prep = sub.add_parser("prepare", help="Milestone 1: prepare legal clause dataset")
    prep.add_argument("--samples-per-class", type=int, default=120)

    sub.add_parser("extend", help="Milestone 2: extend GPT-2 tokenizer and model with custom tags")

    lora_cmd = sub.add_parser("inject-lora", help="Milestone 3: inject LoRA adapters into GPT-2")
    lora_cmd.add_argument("--rank", type=int, default=8)
    lora_cmd.add_argument("--alpha", type=int, default=16)

    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Legal Document LoRA Tagger is ready. Open README.md for your milestones.")
    elif args.command == "prepare":
        stats = milestone_1(samples_per_class=args.samples_per_class)
        print(f"Prepared dataset: {stats}")
    elif args.command == "extend":
        res = milestone_2()
        print(
            f"Extended GPT-2 vocab: {res['orig_vocab_size']} -> {res['new_vocab_size']} | "
            f"Tag IDs: {res['tag_to_id']}"
        )
    elif args.command == "inject-lora":
        res = milestone_3(rank=args.rank, alpha=args.alpha)
        print(
            f"Injected {res['injected_layers']} LoRA layers (rank={res['rank']}, alpha={res['alpha']}) | "
            f"LoRA params: {res['lora_only_params']:,} | "
            f"Total trainable (incl. embeddings): {res['trainable_params']:,} / {res['total_params']:,} "
            f"({res['trainable_pct']}%)"
        )


if __name__ == "__main__":
    main()