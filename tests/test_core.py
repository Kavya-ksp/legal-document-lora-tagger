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
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Implement LoRA adapter and inject into GPT-2
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


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

