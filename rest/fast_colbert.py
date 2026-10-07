"""CPU-optimierter ColBERT(v2)-Reranker.

Liefert dieselben Scores wie rerankers.models.colbert_ranker.ColBERTRanker
(gleiche Query-Kodierung, gleiche MaxSim-Formel), ist aber auf CPU deutlich schneller:

* bf16-Autocast: nutzt AMX auf Intel Sapphire Rapids (m7i/c7i/r7i) für alle Matmuls
* Doc-Länge begrenzt (DOC_MAXLEN, Default 180 = ColBERT-Standard statt 510)
* Dokumente einmal tokenisiert (statt zweimal), Marker-Token ohne Python-Tensor-Schleifen
* Längen-Bucketing: Batches aus ähnlich langen Docs -> kaum Padding-Rechenzeit
* optionaler LRU-Cache für Doc-Embeddings (ColBERT-Late-Interaction: Docs sind
  query-unabhängig, wiederkehrende Docs müssen nicht neu durchs Modell)
"""
from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from math import ceil
from typing import List, Optional

import torch
from transformers import AutoTokenizer

from rerankers.models.colbert_ranker import ColBERTModel


def _dtype(name: str) -> Optional[torch.dtype]:
    name = (name or "").lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    return None  # fp32


class FastColBERT:
    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        doc_maxlen: int = 180,
        batch_size: int = 32,
        dtype: str = "bf16",
        num_threads: Optional[int] = None,
        cache_size: int = 0,
        query_token: str = "[unused0]",
        document_token: str = "[unused1]",
    ):
        if num_threads:
            torch.set_num_threads(num_threads)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ColBERTModel.from_pretrained(model_name, verbose=0).eval()
        self.autocast_dtype = _dtype(dtype)
        self.batch_size = batch_size
        max_pos = self.model.config.max_position_embeddings - 2
        self.doc_maxlen = min(doc_maxlen, max_pos)
        self.query_maxlen = max_pos  # wie Original: Query wird nicht auf 32 gekappt
        self.q_marker = self.tokenizer.convert_tokens_to_ids(query_token)
        self.d_marker = self.tokenizer.convert_tokens_to_ids(document_token)
        self.pad_id = self.tokenizer.pad_token_id
        self.mask_id = self.tokenizer.mask_token_id
        self.cache_size = cache_size
        self._cache: "OrderedDict[bytes, torch.Tensor]" = OrderedDict()
        self._lock = threading.Lock()  # eine Inferenz gleichzeitig: keine Thread-Überbelegung
        self.cache_hits = 0
        self.cache_misses = 0

    # ---------- Tokenisierung ----------
    def _tokenize(self, texts: List[str], maxlen: int, marker: int) -> List[List[int]]:
        ids = self.tokenizer(
            texts, max_length=maxlen - 1, truncation=True, add_special_tokens=True
        )["input_ids"]
        return [seq[:1] + [marker] + seq[1:] for seq in ids]

    def _forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            if self.autocast_dtype is not None:
                with torch.autocast("cpu", dtype=self.autocast_dtype):
                    embs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                embs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embs = embs.float()
        return embs / embs.norm(dim=-1, keepdim=True)

    # ---------- Query ----------
    def encode_query(self, query: str):
        seq = self._tokenize([query], self.query_maxlen, self.q_marker)[0]
        n = len(seq)
        qlen = n + 8 if n % 16 <= 8 else ceil(n / 16) * 16  # identisch zum Original
        ids = seq + [self.mask_id] * (qlen - n)
        mask = [1] * n + [0] * (qlen - n)
        ids_t = torch.tensor([ids])
        mask_t = torch.tensor([mask])
        return self._forward(ids_t, mask_t)[0], n  # [Lq, dim], Anzahl echter Tokens

    # ---------- Dokumente ----------
    @staticmethod
    def _key(text: str) -> bytes:
        return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()

    def _encode_docs(self, texts: List[str]) -> List[torch.Tensor]:
        seqs = self._tokenize(texts, self.doc_maxlen, self.d_marker)
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        out: List[Optional[torch.Tensor]] = [None] * len(seqs)
        for b in range(0, len(order), self.batch_size):
            idx = order[b : b + self.batch_size]
            L = max(len(seqs[i]) for i in idx)
            ids = torch.full((len(idx), L), self.pad_id, dtype=torch.long)
            mask = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                ids[r, : len(seqs[i])] = torch.tensor(seqs[i])
                mask[r, : len(seqs[i])] = 1
            embs = self._forward(ids, mask)
            for r, i in enumerate(idx):
                out[i] = embs[r, : len(seqs[i])].to(torch.float16).clone()
        return out  # type: ignore

    def doc_embeddings(self, texts: List[str]) -> List[torch.Tensor]:
        if self.cache_size <= 0:
            return self._encode_docs(texts)
        keys = [self._key(t) for t in texts]
        res: List[Optional[torch.Tensor]] = [None] * len(texts)
        missing = []
        for i, k in enumerate(keys):
            hit = self._cache.get(k)
            if hit is not None:
                self._cache.move_to_end(k)
                res[i] = hit
                self.cache_hits += 1
            else:
                missing.append(i)
                self.cache_misses += 1
        if missing:
            first: "OrderedDict[bytes, int]" = OrderedDict()
            for i in missing:
                first.setdefault(keys[i], i)
            new = dict(zip(first.keys(), self._encode_docs([texts[i] for i in first.values()])))
            for k, e in new.items():
                self._cache[k] = e
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            for i in missing:
                res[i] = new[keys[i]]
        return res  # type: ignore

    # ---------- Scoring ----------
    @staticmethod
    def maxsim(q: torch.Tensor, n_q: int, docs: List[torch.Tensor]) -> List[float]:
        L = max(d.shape[0] for d in docs)
        D = torch.zeros((len(docs), L, q.shape[1]), dtype=torch.float32)
        valid = torch.zeros((len(docs), L), dtype=torch.bool)
        for i, d in enumerate(docs):
            D[i, : d.shape[0]] = d.float()
            valid[i, : d.shape[0]] = True
        sim = torch.einsum("qd,ntd->nqt", q, D)  # [N, Lq, Ld]
        sim = sim.masked_fill(~valid.unsqueeze(1), -1e4)
        return (sim.max(-1).values.sum(-1) / n_q).tolist()

    def score(self, query: str, docs: List[str]) -> List[float]:
        with self._lock:
            q, n_q = self.encode_query(query)
            d = self.doc_embeddings(docs)
            return self.maxsim(q, n_q, d)

    @classmethod
    def from_env(cls) -> "FastColBERT":
        return cls(
            model_name=os.getenv("COLBERT_MODEL", "colbert-ir/colbertv2.0"),
            doc_maxlen=int(os.getenv("DOC_MAXLEN", "180")),
            batch_size=int(os.getenv("BATCH_SIZE", "32")),
            dtype=os.getenv("COLBERT_DTYPE", "bf16"),
            num_threads=int(os.getenv("TORCH_THREADS", "0")) or None,
            cache_size=int(os.getenv("EMB_CACHE_SIZE", "0")),
        )
