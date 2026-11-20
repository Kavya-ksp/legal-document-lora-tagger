"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import json
import pytest

from app.core import TARGET_TAGS, milestone_1


# Milestone 1: Prepare a dataset of legal clauses for tagging
def test_milestone_1(tmp_path):
    stats = milestone_1(output_dir=tmp_path, samples_per_class=3, use_hf=False)
    assert stats["tags"] == TARGET_TAGS
    assert stats["train"] > 0 and stats["val"] > 0 and stats["test"] > 0
    for split in ("train", "val", "test"):
        lines = (tmp_path / f"{split}.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == stats[split]
        first = json.loads(lines[0])
        assert first["prompt"].startswith("Clause: ")
        assert f"\nTag: <{first['label']}>" in first["prompt"]


# Milestone 2: Extend GPT-2 tokenizer and model for custom tags
def test_milestone_2():
    from app.core import milestone_2

    res = milestone_2()
    tokenizer = res["tokenizer"]
    model = res["model"]

    assert res["new_vocab_size"] == res["orig_vocab_size"] + len(TARGET_TAGS)
    assert model.get_input_embeddings().weight.shape[0] == res["new_vocab_size"]
    for tag, token_id in res["tag_to_id"].items():
        # Each custom <Tag> must encode as a single atomic token ID
        assert tokenizer.encode(tag, add_special_tokens=False) == [token_id]


# Milestone 3: Implement LoRA adapter and inject into GPT-2
def test_milestone_3():
    import torch
    from app.core import LoRAConv1D, milestone_2, milestone_3

    m2 = milestone_2()
    model = m2["model"]
    model.eval()
    sample_ids = torch.tensor([[50257, 100, 200]])
    with torch.no_grad():
        logits_before = model(sample_ids).logits.clone()

    res = milestone_3(model=model, tokenizer=m2["tokenizer"], rank=4, alpha=8)
    assert res["injected_layers"] == 24  # 12 blocks * (c_attn + c_proj)
    assert isinstance(res["model"].transformer.h[0].attn.c_attn, LoRAConv1D)
    assert res["lora_only_params"] < 250_000

    # Because lora_B is zero-initialized, initial forward pass must match base GPT-2 identically
    res["model"].eval()
    with torch.no_grad():
        logits_after = res["model"](sample_ids).logits
    assert torch.allclose(logits_before, logits_after, atol=1e-5)

# Milestone 4: Fine-tune GPT-2 with LoRA on the clause dataset
def test_milestone_4(tmp_path):
    from app.core import milestone_1, milestone_4

    data_dir = tmp_path / "data"
    out_dir = tmp_path / "artifacts"
    milestone_1(output_dir=data_dir, samples_per_class=3, use_hf=False)

    res = milestone_4(
        data_dir=data_dir,
        output_dir=out_dir,
        epochs=1,
        batch_size=4,
        rank=4,
        alpha=8,
        max_steps=3,
    )
    assert res["final_train_loss"] > 0
    assert (out_dir / "lora_adapter.pt").exists()


# Milestone 5: Merge LoRA adapter and perform inference on new clauses
def test_milestone_5(tmp_path):
    from app.core import LoRAConv1D, load_merged_tagger, milestone_1, milestone_4, milestone_5

    data_dir = tmp_path / "data"
    out_dir = tmp_path / "artifacts"
    milestone_1(output_dir=data_dir, samples_per_class=3, use_hf=False)
    milestone_4(data_dir=data_dir, output_dir=out_dir, epochs=1, batch_size=4, rank=4, alpha=8, max_steps=2)

    ckpt_path = out_dir / "lora_adapter.pt"
    bundle = load_merged_tagger(ckpt_path)
    # Confirm LoRAConv1D wrappers were folded back into standard Conv1D layers
    assert not isinstance(bundle["model"].transformer.h[0].attn.c_attn, LoRAConv1D)

    res = milestone_5(
        clauses=["This Agreement is governed by the laws of New York."],
        checkpoint_path=ckpt_path,
    )
    assert res["merged"] is True
    assert res["num_predictions"] == 1
    assert res["predictions"][0]["predicted_tag"] in TARGET_TAGS
    assert 0.0 <= res["predictions"][0]["confidence"] <= 1.0

# Milestone 6: Evaluate tagging accuracy and model efficiency
def test_milestone_6(tmp_path):
    from app.core import milestone_1, milestone_4, milestone_6

    data_dir = tmp_path / "data"
    out_dir = tmp_path / "artifacts"
    report_path = out_dir / "eval_report.json"

    milestone_1(output_dir=data_dir, samples_per_class=3, use_hf=False)
    milestone_4(data_dir=data_dir, output_dir=out_dir, epochs=1, batch_size=4, rank=4, alpha=8, max_steps=2)

    report = milestone_6(
        data_dir=data_dir,
        checkpoint_path=out_dir / "lora_adapter.pt",
        report_path=report_path,
    )
    assert report_path.exists()
    assert 0.0 <= report["accuracy"] <= 1.0
    assert 0.0 <= report["macro_f1"] <= 1.0
    assert report["trainable_param_pct"] < 1.0
    assert report["avg_latency_ms_per_clause"] > 0
    assert set(report["per_class"].keys()) == set(TARGET_TAGS)

