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
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Merge LoRA adapter and perform inference on new clauses
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")


# Milestone 6: Evaluate tagging accuracy and model efficiency
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")

