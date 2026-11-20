"""Command-line entry point: `python -m app.cli --help`."""
import argparse

from app.core import milestone_1, milestone_2, milestone_3, milestone_4, milestone_5, milestone_6


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

    train_cmd = sub.add_parser("train", help="Milestone 4: fine-tune GPT-2 with LoRA on legal clauses")
    train_cmd.add_argument("--epochs", type=int, default=3)
    train_cmd.add_argument("--batch-size", type=int, default=8)
    train_cmd.add_argument("--lr", type=float, default=5e-4)
    train_cmd.add_argument("--rank", type=int, default=8)
    train_cmd.add_argument("--alpha", type=int, default=16)

    tag_cmd = sub.add_parser("tag", help="Milestone 5: merge LoRA adapter and classify legal clauses")
    tag_cmd.add_argument("--clause", type=str, default=None, help="Single custom legal clause to classify")
    tag_cmd.add_argument("--checkpoint", type=str, default="artifacts/lora_adapter.pt")

    eval_cmd = sub.add_parser("evaluate", help="Milestone 6: evaluate tagging accuracy and model efficiency")
    eval_cmd.add_argument("--data-dir", type=str, default="data/processed")
    eval_cmd.add_argument("--checkpoint", type=str, default="artifacts/lora_adapter.pt")
    eval_cmd.add_argument("--report", type=str, default="artifacts/eval_report.json")

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
    elif args.command == "train":
        res = milestone_4(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            rank=args.rank,
            alpha=args.alpha,
        )
        print(f"Training complete: {res}")
    elif args.command == "tag":
        clauses = [args.clause] if args.clause else None
        res = milestone_5(clauses=clauses, checkpoint_path=args.checkpoint)
        for item in res["predictions"]:
            print(f"[{item['predicted_tag']}] ({item['confidence']*100:.1f}%) -> {item['clause']}")
    elif args.command == "evaluate":
        report = milestone_6(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            report_path=args.report,
        )
        print(
            f"Evaluation ({report['test_samples']} test clauses on {report['device']}): "
            f"Accuracy={report['accuracy']*100:.2f}% | Macro-F1={report['macro_f1']:.4f} | "
            f"Trainable Params={report['effective_trainable_params']:,}/{report['total_model_params']:,} "
            f"({report['trainable_param_pct']}%) | "
            f"Latency={report['avg_latency_ms_per_clause']} ms/clause "
            f"({report['throughput_clauses_per_sec']} clauses/s)"
        )
        for tag, metrics in report["per_class"].items():
            print(
                f"  - {tag:<16}: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, "
                f"F1={metrics['f1']:.2f} (n={metrics['support']})"
            )


if __name__ == "__main__":
    main()