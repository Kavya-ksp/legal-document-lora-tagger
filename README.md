# Legal Document LoRA Tagger

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/fuk4wxhzux/badge.svg)](https://ask.thili.ai/b/fuk4wxhzux)

A CLI-based NLP pipeline that fine-tunes GPT-2 (124M parameters) using Low-Rank Adaptation (**LoRA**) to classify legal contract clauses into canonical categories, assisting public sector clerks in reviewing public records faster with minimal compute overhead.

## Completed Milestones

- [x] **1. Prepare a dataset of legal clauses for tagging** — Stratified 960 clauses (120 per class, seed `42`) across 8 legal categories from Hugging Face `lex_glue` (`LEDGAR`) into 80/10/10 train/validation/test splits.
- [x] **2. Extend GPT-2 tokenizer and model for custom tags** — Added 8 atomic `<Tag>` tokens (`<Governing Laws>`, `<Confidentiality>`, `<Termination>`, `<Indemnification>`, `<Notices>`, `<Warranties>`, `<Payments>`, `<Severability>`), expanding vocabulary from `50,257` to `50,265`.
- [x] **3. Implement LoRA adapter and inject into GPT-2** — Built custom `LoRAConv1D` modules (`rank=8`, `alpha=16`) and injected them into all 24 GPT-2 attention projection layers (`c_attn` and `c_proj`).
- [x] **4. Fine-tune GPT-2 with LoRA on the clause dataset** — Trained LoRA weights and new `<Tag>` embedding rows (`0.3604%` of total model parameters) for 3 epochs using masked next-token cross-entropy loss (`val_loss` converged from `0.7693` to `0.0463`).
- [x] **5. Merge LoRA adapter and perform inference on new clauses** — Permanently folded $W_0 + \frac{\alpha}{r}(A \cdot B)$ back into base GPT-2 `Conv1D` weights for zero-overhead inference.
- [x] **6. Evaluate tagging accuracy and model efficiency** — Benchmarked held-out test accuracy, Macro-F1, parameter efficiency, and CPU inference latency, saving structured metrics to `artifacts/eval_report.json`.

## Setup

Requires Python 3.12+ and `git`.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
pytest



## CLI Usage

Run each stage of the pipeline from the project root (with `.venv` activated):

```bash
# Milestone 1: Prepare stratified LEDGAR legal clause dataset
python -m app.cli prepare --samples-per-class 120

# Milestone 2: Extend GPT-2 tokenizer and embeddings with 8 custom <Tag> tokens
python -m app.cli extend

# Milestone 3: Inject LoRAConv1D adapters into GPT-2 attention layers
python -m app.cli inject-lora --rank 8 --alpha 16

# Milestone 4: Fine-tune LoRA adapters and <Tag> embeddings using masked tag loss
python -m app.cli train --epochs 3

# Milestone 5: Merge LoRA weights into base GPT-2 and classify new clauses
python -m app.cli tag
python -m app.cli tag --clause "All disputes shall be resolved under the laws of Texas."

# Milestone 6: Evaluate held-out accuracy, Macro-F1, parameter efficiency, and latency
python -m app.cli evaluate


## Evaluation Results

Evaluated on 96 held-out LEDGAR test clauses (12 per category) on CPU after permanent LoRA weight merging:

### Summary Metrics
- **Test Accuracy:** 77.08%
- **Macro F1-Score:** 0.7439
- **Effective Trainable Parameters:** 448,512 / 124,445,952 (0.3604%)
- **Merged Inference Latency (CPU):** 71.91 ms/clause (13.91 clauses/s)
- **Automated Test Suite (`pytest`):** 9 passed


### Per-Class Performance

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| Governing Laws | 1.00 | 0.67 | 0.80 | 12 |
| Confidentiality | 1.00 | 0.83 | 0.91 | 12 |
| Termination | 0.40 | 1.00 | 0.57 | 12 |
| Indemnification | 1.00 | 0.75 | 0.86 | 12 |
| Notices | 1.00 | 0.92 | 0.96 | 12 |
| Warranties | 1.00 | 1.00 | 1.00 | 12 |
| Payments | 0.75 | 1.00 | 0.86 | 12 |
| Severability | 0.00 | 0.00 | 0.00 | 12 |

### Critical Self-Evaluation

- **Strengths:** With only **0.3604%** of GPT-2's weights updated, the merged model achieves `0.80–1.00` F1 across 6 of the 8 clause categories (`Warranties` at `1.00`, `Notices` at `0.96`, `Confidentiality` at `0.91`) while maintaining fast CPU throughput (`13.91 clauses/s`) with zero runtime adapter overhead.
- **Limitations & Error Analysis:** `Severability` (`0.00` F1) was consistently predicted as `Termination` (`0.40` precision, `1.00` recall). Both clause types share heavy legal boilerplate around contract invalidity, unenforceable provisions, and survival/termination of agreements, and 3 epochs with `rank=8` were insufficient for the newly initialized `<Severability>` embedding to separate from `<Termination>`.
- **Next Steps:** Training for 5+ epochs, increasing LoRA rank (`r=16`), or adding contrastive hard-negative examples between `Severability` and `Termination` would directly resolve this confusion.