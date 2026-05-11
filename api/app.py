import math
import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tiktoken

# ── Hyperparameters (must match the trained model) ──────────────────────────
block_size = 256
n_embd     = 512
n_head     = 8
n_layer    = 8
dropout    = 0.1
vocab_size = 50257   # GPT-2 BPE vocab

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model definition (identical to training code) ────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head  = n_head
        self.head_dim = n_embd // n_head
        self.c_attn  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj  = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
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
        return self.dropout(self.c_proj(out))


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
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs      = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index      = torch.cat((index, index_next), dim=1)
        return index


# ── Load model ───────────────────────────────────────────────────────────────
MODEL_PATH = "../model-owt.pkl"   # adjust path if needed

print(f"Loading model from {MODEL_PATH} on {device}...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
model.eval()
model.to(device)
print(f"Model loaded — {sum(p.numel() for p in model.parameters()):,} parameters")

# ── Tokenizer ────────────────────────────────────────────────────────────────
enc    = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="MyLLM API", description="Text generation API using a GPT trained from scratch on OpenWebText")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt:      str   = "The future of AI is"
    max_tokens:  int   = 200
    temperature: float = 0.8
    top_k:       int   = 40


class GenerateResponse(BaseModel):
    prompt:    str
    generated: str
    full_text: str


@app.get("/")
def root():
    return {
        "model":      "MyLLM GPT",
        "parameters": f"{sum(p.numel() for p in model.parameters()):,}",
        "device":     device,
        "status":     "ready",
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    if req.max_tokens < 1 or req.max_tokens > 500:
        raise HTTPException(status_code=400, detail="max_tokens must be between 1 and 500")

    tokens  = encode(req.prompt)
    context = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    output  = model.generate(
        context,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
    )

    full_text     = decode(output[0].tolist())
    generated_new = decode(output[0].tolist()[len(tokens):])

    return GenerateResponse(
        prompt=req.prompt,
        generated=generated_new,
        full_text=full_text,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
