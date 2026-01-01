import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFTextEmbedder:
    """
    Lightweight text embedder using a Hugging Face encoder model + mean pooling.
    Avoids OpenAI 'text-embedding-ada-002' and avoids sentence-transformers dependency.
    """
    def __init__(self, model_id: str, device: str | None = None, max_length: int = 512):
        self.model_id = model_id
        self.device = device or _default_device()
        self.max_length = max_length # setting max_length to a lower value speeds up embedding but will truncate longer texts, causing information loss and suboptimal retrieval, affecting final performance

        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def embed(self, texts: list[str]) -> np.ndarray:
        # shape: [B, D]
        batch = self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        out = self.model(**batch)
        last_hidden = out.last_hidden_state  # [B, T, D]
        attn = batch["attention_mask"].unsqueeze(-1)  # [B, T, 1]

        summed = (last_hidden * attn).sum(dim=1)      # [B, D]
        denom = attn.sum(dim=1).clamp(min=1)          # [B, 1] # counts non-padded tokens
        mean_pooled = summed / denom                  # [B, D]

        emb = mean_pooled.detach().float().cpu().numpy().astype(np.float32)
        # normalize for cosine
        emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        return emb


class MedPromptMemory:
    """
    Stores:
      - examples: list[dict] (patient fields + model_reasoning + model_answer + labels)
      - embeddings: np.ndarray [N, D] L2-normalized
    Retrieval uses cosine similarity via dot product.
    """
    def __init__(self, embedder: HFTextEmbedder):
        self.embedder = embedder
        self.examples: list[dict] = []
        self.embeddings: np.ndarray | None = None

    def add(self, example: dict, embedding: np.ndarray):
        # embedding: shape [D] or [1, D]
        if embedding.ndim == 2:
            embedding = embedding[0]
        self.examples.append(example)

        if self.embeddings is None:
            self.embeddings = embedding.reshape(1, -1).astype(np.float32)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding.astype(np.float32)])

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        emb_path = os.path.join(out_dir, "embeddings.npy")
        ex_path = os.path.join(out_dir, "examples.jsonl")
        meta_path = os.path.join(out_dir, "meta.json")

        if self.embeddings is None:
            raise ValueError("No embeddings to save (memory is empty).")

        np.save(emb_path, self.embeddings)

        with open(ex_path, "w", encoding="utf-8") as f:
            for ex in self.examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {"embed_model_id": self.embedder.model_id, "n": len(self.examples), "dim": int(self.embeddings.shape[1])},
                f,
                indent=2,
            )

    @staticmethod
    def load(memory_dir: str, embedder: HFTextEmbedder):
        emb_path = os.path.join(memory_dir, "embeddings.npy")
        ex_path = os.path.join(memory_dir, "examples.jsonl")

        if not os.path.exists(emb_path) or not os.path.exists(ex_path):
            raise FileNotFoundError(f"MedPrompt memory not found in: {memory_dir}")

        mem = MedPromptMemory(embedder=embedder)
        mem.embeddings = np.load(emb_path).astype(np.float32)

        # ensure normalized
        mem.embeddings /= (np.linalg.norm(mem.embeddings, axis=1, keepdims=True) + 1e-12)

        with open(ex_path, "r", encoding="utf-8") as f:
            mem.examples = [json.loads(line) for line in f if line.strip()]

        return mem

    def retrieve(self, query_text: str, k: int = 4) -> list[dict]:
        if self.embeddings is None or len(self.examples) == 0 or k <= 0:
            return []

        q = self.embedder.embed([query_text])[0]  # [D], normalized
        sims = self.embeddings @ q                # [N]
        k = min(k, sims.shape[0])

        # k could become 0 if sims.shape[0] == 0 (defensive), keep safe:
        if k <= 0:
            return []

        idx = np.argpartition(-sims, kth=k-1)[:k]
        idx = idx[np.argsort(-sims[idx])]         # sort best->worst
        return [self.examples[int(i)] for i in idx]