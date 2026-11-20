"""Your code. One stub per milestone; build them in order and keep each one small."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time

TARGET_TAGS = [
    "Governing Laws",
    "Confidentiality",
    "Termination",
    "Indemnification",
    "Notices",
    "Warranties",
    "Payments",
    "Severability",
]


def tag_token(label: str) -> str:
    return f"<{label}>"


def format_training_example(clause: str, label: str) -> str:
    clean_clause = " ".join(clause.strip().split())
    return f"Clause: {clean_clause}\nTag: {tag_token(label)}"


def _fallback_records() -> list[dict[str, str]]:
    samples = {
        "Governing Laws": [
            "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware.",
            "All disputes arising under this Contract shall be governed by the internal laws of the State of New York.",
            "The validity and interpretation of this Agreement shall be governed by the laws of England and Wales.",
        ],
        "Confidentiality": [
            "Each party agrees to hold all Confidential Information in strict confidence and not disclose it to third parties.",
            "Recipient shall protect Proprietary Information using the same degree of care it uses for its own secrets.",
            "Confidential Information shall not be used for any purpose outside the performance of this Agreement.",
        ],
        "Termination": [
            "Either party may terminate this Agreement upon thirty (30) days prior written notice to the other party.",
            "Company may terminate this Contract immediately upon written notice if Supplier commits a material breach.",
            "Upon termination of this Agreement, all licenses granted hereunder shall immediately cease.",
        ],
        "Indemnification": [
            "Vendor shall indemnify, defend, and hold harmless Customer from any third-party claims arising from negligence.",
            "Each party agrees to indemnify the other against all losses, damages, and reasonable attorney fees.",
            "Seller shall indemnify Buyer for any bodily injury or property damage caused by defective Products.",
        ],
        "Notices": [
            "All notices required under this Agreement shall be in writing and delivered to the addresses set forth herein.",
            "Any notice given by certified mail shall be deemed received three (3) business days after mailing.",
            "Notices sent by email with confirmation of receipt shall constitute valid written notice under this Contract.",
        ],
        "Warranties": [
            "Supplier warrants that all Goods delivered shall be free from defects in material and workmanship for one year.",
            "EXCEPT AS EXPRESSLY SET FORTH HEREIN, SELLER DISCLAIMS ALL IMPLIED WARRANTIES OF MERCHANTABILITY.",
            "Contractor represents and warrants that Services will be performed in a professional and workmanlike manner.",
        ],
        "Payments": [
            "Customer shall pay all undisputed invoices within thirty (30) days of the invoice date in US Dollars.",
            "Late payments shall accrue interest at the rate of 1.5% per month until paid in full.",
            "All fees are non-refundable and exclusive of applicable sales, use, or value-added taxes.",
        ],
        "Severability": [
            "If any provision of this Agreement is held invalid or unenforceable, the remaining provisions remain in full force.",
            "The invalidity of any clause shall not affect the validity of any other provision of this Contract.",
            "Any unenforceable term shall be reformed to the minimum extent necessary to make it valid and enforceable.",
        ],
    }
    records = []
    for label, texts in samples.items():
        for text in texts:
            records.append({"text": text, "label": label, "prompt": format_training_example(text, label)})
    return records


# --- Milestone 1: Prepare a dataset of legal clauses for tagging
def milestone_1(
    output_dir: str | Path = "data/processed",
    samples_per_class: int = 120,
    use_hf: bool = True,
) -> dict[str, Any]:
    """Load, balance, format, and save stratified train/val/test legal clause splits."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    canonical_by_lower = {
        "governing laws": "Governing Laws",
        "confidentiality": "Confidentiality",
        "termination": "Termination",
        "terminations": "Termination",
        "indemnification": "Indemnification",
        "indemnifications": "Indemnification",
        "notices": "Notices",
        "warranties": "Warranties",
        "payments": "Payments",
        "severability": "Severability",
    }
    by_label: dict[str, list[dict[str, str]]] = {label: [] for label in TARGET_TAGS}

    if use_hf:
        try:
            from datasets import load_dataset

            ds = load_dataset("coastalcph/lex_glue", "ledgar", split="train")
            label_names = ds.features["label"].names
            for row in ds:
                raw_lbl = label_names[row["label"]].lower()
                if raw_lbl in canonical_by_lower:
                    lbl = canonical_by_lower[raw_lbl]
                    if len(by_label[lbl]) < samples_per_class:
                        text = " ".join(row["text"].strip().split())
                        if 30 <= len(text) <= 900:
                            by_label[lbl].append(
                                {"text": text, "label": lbl, "prompt": format_training_example(text, lbl)}
                            )
                if all(len(items) >= samples_per_class for items in by_label.values()):
                    break
        except Exception:
            by_label = {label: [] for label in TARGET_TAGS}

    if not all(len(items) >= 3 for items in by_label.values()):
        by_label = {label: [] for label in TARGET_TAGS}
        for rec in _fallback_records():
            by_label[rec["label"]].append(rec)

    train_rows, val_rows, test_rows = [], [], []
    for label in TARGET_TAGS:
        items = by_label[label]
        n = len(items)
        if n == 3:
            n_train, n_val = 1, 1
        else:
            n_train = max(1, int(0.8 * n))
            n_val = max(1, int(0.1 * n))
        train_rows.extend(items[:n_train])
        val_rows.extend(items[n_train : n_train + n_val])
        test_rows.extend(items[n_train + n_val :])

    splits = {"train": train_rows, "val": val_rows, "test": test_rows}
    for split_name, rows in splits.items():
        with (out_path / f"{split_name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    (out_path / "tags.json").write_text(json.dumps(TARGET_TAGS, indent=2), encoding="utf-8")
    return {
        "tags": TARGET_TAGS,
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
        "output_dir": str(out_path),
    }


# --- Milestone 2: Extend GPT-2 tokenizer and model for custom tags
def milestone_2(
    model_name: str = "gpt2",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Extend GPT-2 tokenizer and token embeddings with custom atomic <Tag> tokens."""
    active_tags = tags or TARGET_TAGS
    special_tags = [tag_token(label) for label in active_tags]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    orig_vocab_size = len(tokenizer)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tags})
    new_vocab_size = len(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(new_vocab_size)

    tag_to_id = {tag: tokenizer.convert_tokens_to_ids(tag) for tag in special_tags}
    return {
        "tokenizer": tokenizer,
        "model": model,
        "tag_tokens": special_tags,
        "tag_to_id": tag_to_id,
        "orig_vocab_size": orig_vocab_size,
        "new_vocab_size": new_vocab_size,
    }

class LoRAConv1D(nn.Module):
    """Low-Rank Adaptation wrapper for GPT-2's Conv1D layers (weight shape: in_features x out_features)."""

    def __init__(self, base_layer: nn.Module, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.base_layer = base_layer
        in_features, out_features = base_layer.weight.shape
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


# --- Milestone 3: Implement LoRA adapter and inject into GPT-2
def milestone_3(
    model: nn.Module | None = None,
    tokenizer: Any | None = None,
    rank: int = 8,
    alpha: int = 16,
    target_c_proj: bool = True,
) -> dict[str, Any]:
    """Freeze base GPT-2, inject LoRAConv1D into attention layers, and keep tag embeddings trainable."""
    if model is None or tokenizer is None:
        m2 = milestone_2()
        model = m2["model"]
        tokenizer = m2["tokenizer"]

    # 1. Freeze all base parameters first
    for param in model.parameters():
        param.requires_grad = False

    # 2. Inject LoRAConv1D into every transformer block's attention layers
    injected_layers = 0
    for block in model.transformer.h:
        if not isinstance(block.attn.c_attn, LoRAConv1D):
            block.attn.c_attn = LoRAConv1D(block.attn.c_attn, rank=rank, alpha=alpha)
            injected_layers += 1
        if target_c_proj and not isinstance(block.attn.c_proj, LoRAConv1D):
            block.attn.c_proj = LoRAConv1D(block.attn.c_proj, rank=rank, alpha=alpha)
            injected_layers += 1

    # 3. Unfreeze token embeddings so the newly added <Tag> rows can learn
    model.transformer.wte.weight.requires_grad = True

    trainable_params, total_params = count_parameters(model)
    lora_only_params = sum(
        p.numel() for name, p in model.named_parameters() if p.requires_grad and "lora_" in name
    )
    return {
        "model": model,
        "tokenizer": tokenizer,
        "rank": rank,
        "alpha": alpha,
        "injected_layers": injected_layers,
        "lora_only_params": lora_only_params,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": round(100.0 * trainable_params / total_params, 4),
    }


class ClauseTagDataset(Dataset):
    """Tokenizes legal clauses and masks loss so gradients only flow from the target <Tag> token."""

    def __init__(self, rows: list[dict[str, str]], tokenizer: Any, max_length: int = 192):
        self.samples: list[tuple[list[int], list[int]]] = []
        eos_id = tokenizer.eos_token_id

        for row in rows:
            clean_clause = " ".join(row["text"].strip().split())
            prefix = f"Clause: {clean_clause}\nTag:"
            target_str = f" {tag_token(row['label'])}"

            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            target_ids = tokenizer.encode(target_str, add_special_tokens=False) + [eos_id]

            max_prefix = max_length - len(target_ids)
            if len(prefix_ids) > max_prefix:
                prefix_ids = prefix_ids[:max_prefix]

            input_ids = prefix_ids + target_ids
            labels = ([-100] * len(prefix_ids)) + target_ids
            self.samples.append((input_ids, labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self.samples[idx]


def _collate_clause_batch(batch: list[tuple[list[int], list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(x[0]) for x in batch)
    input_ids, attention_mask, labels = [], [], []
    for ids, lbls in batch:
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        labels.append(lbls + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _load_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- Milestone 4: Fine-tune GPT-2 with LoRA on the clause dataset
def milestone_4(
    data_dir: str | Path = "data/processed",
    output_dir: str | Path = "artifacts",
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 5e-4,
    rank: int = 8,
    alpha: int = 16,
    max_length: int = 192,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Fine-tune LoRA-injected GPT-2 using masked loss on <Tag> tokens and save adapter checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    m2 = milestone_2()
    orig_vocab_size = m2["orig_vocab_size"]
    m3 = milestone_3(model=m2["model"], tokenizer=m2["tokenizer"], rank=rank, alpha=alpha)
    model = m3["model"].to(device)
    tokenizer = m3["tokenizer"]

    # Freeze base vocabulary rows (0..orig_vocab_size-1) so only new <Tag> embeddings update
    def _mask_base_vocab_grad(grad: torch.Tensor) -> torch.Tensor:
        masked = grad.clone()
        masked[:orig_vocab_size] = 0.0
        return masked

    hook_handle = model.transformer.wte.weight.register_hook(_mask_base_vocab_grad)

    train_rows = _load_jsonl(data_path / "train.jsonl")
    val_rows = _load_jsonl(data_path / "val.jsonl")

    train_ds = ClauseTagDataset(train_rows, tokenizer, max_length=max_length)
    val_ds = ClauseTagDataset(val_rows, tokenizer, max_length=max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate_clause_batch(b, tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: _collate_clause_batch(b, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)

    history: list[dict[str, float]] = []
    global_step = 0

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            total_train_loss = 0.0
            train_batches = 0

            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_train_loss += loss.item()
                train_batches += 1
                global_step += 1
                if max_steps is not None and global_step >= max_steps:
                    break

            model.eval()
            total_val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    total_val_loss += model(**batch).loss.item()
                    val_batches += 1

            avg_train = total_train_loss / max(1, train_batches)
            avg_val = total_val_loss / max(1, val_batches)
            history.append({"epoch": epoch, "train_loss": round(avg_train, 4), "val_loss": round(avg_val, 4)})
            print(f"Epoch {epoch}/{epochs} | train_loss={avg_train:.4f} | val_loss={avg_val:.4f}")

            if max_steps is not None and global_step >= max_steps:
                break
    finally:
        hook_handle.remove()

    # Save only LoRA weights + newly learned <Tag> embedding rows
    lora_state = {
        name: param.detach().cpu()
        for name, param in model.state_dict().items()
        if "lora_" in name
    }
    tag_embeddings = model.transformer.wte.weight[orig_vocab_size:].detach().cpu()

    checkpoint_path = out_path / "lora_adapter.pt"
    torch.save(
        {
            "rank": rank,
            "alpha": alpha,
            "orig_vocab_size": orig_vocab_size,
            "tags": TARGET_TAGS,
            "lora_state_dict": lora_state,
            "tag_embeddings": tag_embeddings,
            "history": history,
        },
        checkpoint_path,
    )

    return {
        "device": str(device),
        "epochs_completed": len(history),
        "initial_train_loss": history[0]["train_loss"],
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size_kb": round(checkpoint_path.stat().st_size / 1024, 1),
    }


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Fold LoRAConv1D matrices (W_0 + scaling * A @ B) back into standard base Conv1D layers."""
    for block in model.transformer.h:
        for attr in ("c_attn", "c_proj"):
            layer = getattr(block.attn, attr)
            if isinstance(layer, LoRAConv1D):
                with torch.no_grad():
                    delta = (layer.lora_A @ layer.lora_B) * layer.scaling
                    layer.base_layer.weight.add_(delta.to(layer.base_layer.weight.dtype))
                setattr(block.attn, attr, layer.base_layer)
    return model


def load_merged_tagger(checkpoint_path: str | Path = "artifacts/lora_adapter.pt") -> dict[str, Any]:
    """Load trained LoRA adapter + tag embeddings, inject, restore weights, and merge into base GPT-2."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    tags = ckpt["tags"]
    orig_vocab_size = ckpt["orig_vocab_size"]

    m2 = milestone_2(tags=tags)
    m3 = milestone_3(
        model=m2["model"],
        tokenizer=m2["tokenizer"],
        rank=ckpt["rank"],
        alpha=ckpt["alpha"],
    )
    model = m3["model"]
    tokenizer = m3["tokenizer"]

    # Restore trained LoRA matrices and the 8 learned <Tag> token embedding rows
    model.load_state_dict(ckpt["lora_state_dict"], strict=False)
    with torch.no_grad():
        model.transformer.wte.weight[orig_vocab_size:].copy_(ckpt["tag_embeddings"])

    # Permanently fold LoRA weights into base Conv1D layers for zero-overhead inference
    model = merge_lora_weights(model)
    model.eval()

    tag_to_id = {label: tokenizer.convert_tokens_to_ids(tag_token(label)) for label in tags}
    return {
        "model": model,
        "tokenizer": tokenizer,
        "tags": tags,
        "tag_to_id": tag_to_id,
    }


def predict_clause_tag(
    clause: str,
    model: nn.Module,
    tokenizer: Any,
    tag_to_id: dict[str, int],
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Score the 8 atomic <Tag> tokens on the next-token distribution and return the top tag + probabilities."""
    active_device = device or next(model.parameters()).device
    clean_clause = " ".join(clause.strip().split())
    prompt = f"Clause: {clean_clause}\nTag:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=184).to(active_device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]

    ordered_labels = list(tag_to_id.keys())
    ordered_ids = [tag_to_id[lbl] for lbl in ordered_labels]
    tag_logits = logits[ordered_ids]
    probs = torch.softmax(tag_logits, dim=-1).cpu().tolist()

    scores = {lbl: round(float(p), 4) for lbl, p in zip(ordered_labels, probs)}
    best_label = max(scores, key=scores.get)
    return {
        "clause": clean_clause,
        "predicted_tag": best_label,
        "confidence": scores[best_label],
        "scores": dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True)),
    }


# --- Milestone 5: Merge LoRA adapter and perform inference on new clauses
def milestone_5(
    clauses: list[str] | None = None,
    checkpoint_path: str | Path = "artifacts/lora_adapter.pt",
) -> dict[str, Any]:
    """Merge the trained LoRA adapter into GPT-2 and classify new unseen legal clauses."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_merged_tagger(checkpoint_path=checkpoint_path)
    model = bundle["model"].to(device)
    tokenizer = bundle["tokenizer"]
    tag_to_id = bundle["tag_to_id"]

    sample_clauses = clauses or [
        "This Agreement shall be governed by and construed in accordance with the laws of the State of California.",
        "Recipient shall maintain all Proprietary Information in strict confidence for five (5) years.",
        "Either party may terminate this Agreement upon thirty (30) days written notice.",
        "Supplier shall indemnify and hold harmless Buyer from all third-party claims and damages.",
        "All notices under this Contract must be sent in writing via certified mail.",
        "Vendor warrants that the Software will perform materially in accordance with the documentation.",
        "Client shall pay all undisputed invoices within thirty (30) days of receipt.",
        "If any provision of this Agreement is found invalid, the remaining provisions remain enforceable.",
    ]

    predictions = [
        predict_clause_tag(clause, model=model, tokenizer=tokenizer, tag_to_id=tag_to_id, device=device)
        for clause in sample_clauses
    ]
    return {
        "merged": True,
        "num_predictions": len(predictions),
        "predictions": predictions,
    }


# --- Milestone 6: Evaluate tagging accuracy and model efficiency
def milestone_6(
    data_dir: str | Path = "data/processed",
    checkpoint_path: str | Path = "artifacts/lora_adapter.pt",
    report_path: str | Path = "artifacts/eval_report.json",
) -> dict[str, Any]:
    """Evaluate held-out test accuracy, macro F1, parameter efficiency, and inference latency."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    bundle = load_merged_tagger(checkpoint_path=checkpoint_path)

    model = bundle["model"].to(device)
    tokenizer = bundle["tokenizer"]
    tags = bundle["tags"]
    tag_to_id = bundle["tag_to_id"]

    test_rows = _load_jsonl(Path(data_dir) / "test.jsonl")

    correct = 0
    tp = {tag: 0 for tag in tags}
    fp = {tag: 0 for tag in tags}
    fn = {tag: 0 for tag in tags}

    # Warm-up pass before timing
    if test_rows:
        predict_clause_tag(test_rows[0]["text"], model=model, tokenizer=tokenizer, tag_to_id=tag_to_id, device=device)

    start_time = time.perf_counter()
    for row in test_rows:
        pred = predict_clause_tag(
            row["text"], model=model, tokenizer=tokenizer, tag_to_id=tag_to_id, device=device
        )["predicted_tag"]
        gold = row["label"]
        if pred == gold:
            correct += 1
            tp[gold] += 1
        else:
            fp[pred] += 1
            fn[gold] += 1
    elapsed_sec = max(1e-6, time.perf_counter() - start_time)

    per_class: dict[str, dict[str, float]] = {}
    f1_scores: list[float] = []
    for tag in tags:
        prec = tp[tag] / max(1, tp[tag] + fp[tag])
        rec = tp[tag] / max(1, tp[tag] + fn[tag])
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[tag] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": tp[tag] + fn[tag],
        }
        f1_scores.append(f1)

    total_params = sum(p.numel() for p in model.parameters())
    lora_params = sum(v.numel() for v in ckpt["lora_state_dict"].values())
    new_tag_emb_params = int(ckpt["tag_embeddings"].numel())
    effective_trainable_params = lora_params + new_tag_emb_params

    num_samples = len(test_rows)
    accuracy = correct / max(1, num_samples)
    macro_f1 = sum(f1_scores) / max(1, len(f1_scores))
    avg_latency_ms = (elapsed_sec / max(1, num_samples)) * 1000.0
    throughput = num_samples / elapsed_sec

    report = {
        "device": str(device),
        "test_samples": num_samples,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "total_model_params": total_params,
        "lora_params": lora_params,
        "new_tag_embedding_params": new_tag_emb_params,
        "effective_trainable_params": effective_trainable_params,
        "trainable_param_pct": round(100.0 * effective_trainable_params / total_params, 4),
        "avg_latency_ms_per_clause": round(avg_latency_ms, 2),
        "throughput_clauses_per_sec": round(throughput, 2),
        "per_class": per_class,
    }

    out_report = Path(report_path)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report