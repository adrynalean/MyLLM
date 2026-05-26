# CorpusForge — A Large Language Model Trained from Scratch

A GPT-style language model built and trained from the ground up on OpenWebText — the same dataset used to train GPT-2.

---

## What We Built

A decoder-only transformer (GPT architecture) trained entirely from scratch in Python using PyTorch. No pretrained weights, no fine-tuning — every parameter learned from raw web text.

---

## Model Architecture

| Component | Detail |
|---|---|
| Architecture | Decoder-only Transformer (GPT-style) |
| Attention | Multi-head Causal Self-Attention (fused QKV) |
| Attention implementation | `F.scaled_dot_product_attention` (Flash Attention when available) |
| Normalization | Pre-LayerNorm (GPT-2 style, more stable) |
| Activation | GELU |
| Weight tying | Token embedding and output projection (reduces params, improves performance) |
| Initialization | GPT-2 style scaled residual projections |

### Version History

| Version | n_embd | n_head | n_layer | Parameters | Dataset |
|---|---|---|---|---|---|
| v1 | 128 | 4 | 3 | ~1M | Wizard of Oz (50k chars, character-level) |
| v2 | 384 | 6 | 6 | ~30M | OpenWebText (50M tokens, BPE) |
| v3 (current) | 512 | 8 | 8 | ~54M | OpenWebText (50M tokens, BPE) |

---

## Dataset

**OpenWebText** — an open-source replication of OpenAI's WebText dataset (used to train GPT-2).

- **Source:** Reddit outbound links filtered for quality
- **Full size:** 8,013,769 documents (~38GB of text)
- **We used:** ~50 million tokens streamed via HuggingFace `datasets`
- **License:** CC0 1.0

---

## Tokenizer

- **v1:** Character-level (~70 unique tokens) — simple but inefficient
- **v2/v3:** GPT-2 BPE via `tiktoken` (vocab size: 50,257) — same tokenizer used by OpenAI for GPT-2 and GPT-3

---

## Training Setup

| Hyperparameter | v2 (30M) | v3 (54M) |
|---|---|---|
| Batch size | 32 | 32 |
| Block size (context) | 256 tokens | 256 tokens |
| Learning rate | 3e-4 (cosine decay) | 3e-4 (cosine decay) |
| LR warmup | 200 steps | 200 steps |
| Optimizer | AdamW (b1=0.9, b2=0.95) | AdamW (b1=0.9, b2=0.95) |
| Gradient clipping | 1.0 | 1.0 |
| Dropout | 0.1 | 0.1 |
| Hardware | Kaggle T4 GPU (16GB) | Kaggle T4 GPU (16GB) |

---

## Training Results (v2 — 30M params)

| Step | Train Loss | Val Loss |
|---|---|---|
| 0 | 10.89 | 10.89 |
| 500 | 6.22 | 6.22 |
| 1000 | 5.85 | 5.88 |
| 2000 | 5.47 | 5.50 |
| 3000 | 5.31 | 5.35 |
| 4000 | 5.19 | 5.25 |
| 5000 | 5.18 | 5.22 |

---

## Sample Output (v2 — after 5000 steps)

**Prompt:** `The quick brown fox`

> The quick brown fox, the "Bogic" is a great, so we look like to me, and it's better.
>
> The great reason you're the only one of your way of your body. You can find a little less of your mind.
>
> This is why, for example, what I want to do is that all of the same thing, because it's just a bit of a few...

---

## Project Structure

```
MyLLM/
├── v1.ipynb                # Character-level GPT on Wizard of Oz
├── v2_openwebtext.ipynb    # Upgraded: tiktoken + OpenWebText streaming
├── myllm.ipynb             # Kaggle training notebook (current model)
├── bigram.ipynb            # Bigram baseline model (starting point)
├── torch-examples.ipynb    # PyTorch fundamentals experiments
├── api/
│   ├── app.py              # FastAPI inference server
│   └── requirements.txt    # API dependencies
├── vocab.txt               # Character vocabulary (v1)
├── wizard_of_oz.txt        # Training text (v1)
└── .gitignore
```

---

## API

A FastAPI server for text generation using the trained model. See [`api/`](api/) for setup and usage.

```bash
POST /generate
{"prompt": "The future of AI is", "max_tokens": 200}
```

---

## How to Run

### Train (Kaggle)
1. Upload `myllm.ipynb` to Kaggle
2. Enable GPU: Session options → T4 GPU
3. Run All

### API (local)
```bash
cd api
pip install -r requirements.txt
uvicorn app:app --reload
```

---

## Key Learnings

- Built a GPT from scratch — attention, feedforward, positional embeddings, all by hand
- Character-level vs BPE tokenization — BPE is ~10x more efficient for real text
- Cosine LR schedule with warmup stabilizes training
- Weight tying reduces parameters while improving performance
- Flash Attention works out of the box in PyTorch 2.0+ via `scaled_dot_product_attention`
- Streaming large datasets with HuggingFace avoids downloading 55GB upfront
