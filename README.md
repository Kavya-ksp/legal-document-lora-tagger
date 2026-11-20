# Legal Document LoRA Tagger

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/fuk4wxhzux/badge.svg)](https://ask.thili.ai/b/fuk4wxhzux)

This project develops a Legal Document LoRA Tagger, a tool designed to assist public sector clerks. By fine-tuning GPT-2 with LoRA, it automates the classification of legal clauses in public records, enhancing efficiency and reducing the need for manual review.

## What you are building

**Goal.** You will build a tool that uses LoRA to fine-tune GPT-2 to tag legal clauses in public records, resulting in a tagged legal document.

**Done when.** The project is complete when you have a fine-tuned GPT-2 model with a LoRA adapter that can classify legal contract clauses by type, and you have performed inference to produce a tagged legal document.

## Milestones

Tick these off as you go, and commit after each one.

- [X] **1. Prepare a dataset of legal clauses for tagging**  
      Hint: Prepare your dataset, ensuring legal clauses are ready for tagging.
- [X] **2. Extend GPT-2 tokenizer and model for custom tags**  
      Hint: Extend the GPT-2 tokenizer and model to include your custom tag tokens, similar to the `lfllm-lflegalgpt` drill.
- [X] **3. Implement LoRA adapter and inject into GPT-2**  
      Hint: Implement the LoRA adapter and inject it into GPT-2's attention layers, as shown in the `lfllm-lflora` drill.
- [X] **4. Fine-tune GPT-2 with LoRA on the clause dataset**  
      Hint: Fine-tune GPT-2 using the LoRA adapter on your prepared clause dataset, applying techniques from `lfllm-lflegalgpt` and `lfllm-lflora`.
- [X] **5. Merge LoRA adapter and perform inference on new clauses**  
      Hint: Merge the LoRA adapter weights back into the base GPT-2 model and perform inference on new legal clauses.
- [X] **6. Evaluate tagging accuracy and model efficiency**  
      Hint: Evaluate the accuracy of your tagging and the efficiency improvements gained from using LoRA.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Experiment with different LoRA ranks or target layers to optimize performance.
- Implement a user interface for easier document input and tagged output viewing.
- Extend the model to recognize and tag a wider variety of legal clause types beyond the initial dataset.
