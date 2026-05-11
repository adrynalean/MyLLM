import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F
from huggingface_hub import hf_hub_download
import tiktoken
import gradio as gr

# ── Hyperparameters (must match trained model) ───────────────────────────────
block_size = 256
n_embd     = 512
n_head     = 8
n_layer    = 8
dropout    = 0.1
vocab_size = 50257

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ── Model definition ─────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head   = n_head
        self.head_dim = n_embd // n_head
        self.c_attn   = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj   = nn.Linear(n_embd, n_embd, bias=False)
        self.drop     = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.c_proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2  = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.token_embedding_table.weight = self.lm_head.weight

    def forward(self, index, targets=None):
        B, T = index.shape
        x = self.token_embedding_table(index) + self.position_embedding_table(
            torch.arange(T, device=device)
        )
        x = self.ln_f(self.blocks(x))
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, index, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, _  = self.forward(index_cond)
            logits     = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs      = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index      = torch.cat((index, index_next), dim=1)
        return index


# ── Custom unpickler ─────────────────────────────────────────────────────────
class ModelUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        mapping = {
            "GPTLanguageModel":    GPTLanguageModel,
            "CausalSelfAttention": CausalSelfAttention,
            "FeedForward":         FeedForward,
            "Block":               Block,
        }
        if name in mapping:
            return mapping[name]
        return super().find_class(module, name)


# ── Load model from HuggingFace Hub ─────────────────────────────────────────
print("Downloading model from HuggingFace Hub...")
model_path = hf_hub_download(repo_id="Fluoron/MyLLM", filename="model-ow_best.pkl")
print(f"Loading model on {device}...")
with open(model_path, "rb") as f:
    model = ModelUnpickler(f).load()
model.eval()
model.to(device)
print(f"Ready — {sum(p.numel() for p in model.parameters()):,} parameters")

# ── Tokenizer ────────────────────────────────────────────────────────────────
enc    = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)


# ── Gradio inference function ────────────────────────────────────────────────
def generate_text(prompt, max_tokens, temperature, top_k):
    if not prompt.strip():
        return "Please enter a prompt."
    tokens  = encode(prompt)
    context = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    output  = model.generate(context, max_new_tokens=int(max_tokens),
                              temperature=temperature, top_k=int(top_k))
    return decode(output[0].tolist())


# ── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(title="MyLLM — GPT Trained from Scratch") as demo:
    gr.Markdown(
        """
        # MyLLM — GPT Language Model Trained from Scratch
        A **51M parameter** decoder-only transformer trained on **50 million tokens** from
        [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) — the same dataset used to train GPT-2.
        Built entirely from scratch with PyTorch. No pretrained weights.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            prompt_box = gr.Textbox(
                label="Prompt",
                placeholder="Type something to continue...",
                lines=3,
                value="The future of artificial intelligence is"
            )
            with gr.Row():
                max_tokens  = gr.Slider(10, 400, value=200, step=10,  label="Max tokens")
                temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
                top_k       = gr.Slider(1, 100,  value=40,  step=1,   label="Top-k")
            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=3):
            output_box = gr.Textbox(label="Generated text", lines=12, interactive=False)

    generate_btn.click(
        fn=generate_text,
        inputs=[prompt_box, max_tokens, temperature, top_k],
        outputs=output_box,
    )

    gr.Markdown(
        """
        ---
        **Temperature** — higher = more creative/random, lower = more focused
        **Top-k** — limits sampling to the k most likely next tokens
        **Source:** [GitHub](https://github.com/adrynalean/MyLLM)
        """
    )

demo.launch()
