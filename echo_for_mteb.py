from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Optional: only used if the environment provides it (MTEB)
try:
    from mteb.encoder_interface import PromptType
except Exception:

    class PromptType:
        query = "query"
        passage = "passage"


# -----------------------
# EchoParser
# -----------------------
class EchoParser(nn.Module):
    """
    Turn (template_key, sample) pairs into batched tensors:
      - input_ids: (B, L)
      - attention_mask: (B, L)
      - embed_mask: (B, L)  -> 1 only for echo span tokens
    Template grammar:
      - Curly braces `{ ... }` denote a "dynamic piece" processed at runtime.
      - Inside such a piece, `%%key%%` placeholders will be replaced from the sample dict (e.g., {'x': 'text'}).
      - If a dynamic piece string starts with '!', that piece is marked as non-pooling (embed_mask=0).
      - Any text outside `{ ... }` is considered a static piece and NEVER participates in pooling (embed_mask=0).
    Example:
      "<s>{!Q: %%x%%}\n{A: %%x%%}</s>"
      -> first piece non-pooling; second piece pooling.
    """

    def __init__(
        self,
        tokenizer: Union[str, Any],
        templates: Dict[str, str],
        max_length: Optional[int] = 600,
        piece_max_tokens: Optional[int] = 256,
        pad_to_multiple_of: Optional[int] = 8,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.piece_max_tokens = piece_max_tokens
        self.pad_to_multiple_of = pad_to_multiple_of

        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer, add_bos_token=False, add_eos_token=False
            )

        # Stable right-padding + pad_token
        if getattr(self.tokenizer, "padding_side", None) != "right":
            self.tokenizer.padding_side = "right"
        if getattr(self.tokenizer, "pad_token", None) is None:
            # Fallback to unk if pad not set
            try:
                self.tokenizer.pad_token = self.tokenizer.unk_token or "<unk>"
            except Exception:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

        self.templates = templates
        self.template_pieces: Dict[str, List[Union[str, List[int]]]] = {
            k: self._parse_template(v) for k, v in templates.items()
        }
        self.piece_max_tokens = piece_max_tokens

        # ---- Length stats (off by default) ----
        self._collect_len_stats = False
        self._row_stats: List[Dict[str, int]] = []

    # ---- controls for length stats ----
    def enable_length_stats(self, flag: bool = True):
        self._collect_len_stats = flag

    def pop_length_stats(self) -> List[Dict[str, int]]:
        out = self._row_stats
        self._row_stats = []
        return out

    # --- template parsing ---
    def _parse_template(self, template: str) -> List[Union[str, List[int]]]:
        """
        Split template into pieces:
          - text outside { } -> tokenize now (static), embed_mask=0 later
          - text inside { }  -> keep as string for runtime substitution/masking
        """
        pieces: List[Union[str, List[int]]] = []
        idx = 0
        while idx < len(template):
            if template[idx] == "{":
                # find matching }
                end = template.find("}", idx + 1)
                if end == -1:
                    raise ValueError("Unmatched '{' in template")
                inner = template[idx + 1 : end]
                pieces.append(inner)  # keep as string (dynamic piece)
                idx = end + 1
            else:
                # accumulate until next { or end
                next_brace = template.find("{", idx)
                if next_brace == -1:
                    literal = template[idx:]
                    idx = len(template)
                else:
                    literal = template[idx:next_brace]
                    idx = next_brace
                if literal:
                    ids = self.tokenizer(literal, add_special_tokens=False)["input_ids"]
                    pieces.append(ids)  # static piece -> already tokenized
        return pieces

    # 2) 工具函式：用 tokenizer 依 token 數截斷一段文字
    def _clamp_by_tokens(self, text: str, max_tokens: int) -> str:
        # 只截斷內容本身，不要讓 tokenizer 自動加 special tokens
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        ids = ids[:max_tokens]
        # 用 decode 還原；避免清理格式導致多餘變動
        return self.tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    # 3) 在 _tokenize_dynamic_piece 裡，替換 %%key%% 前先對 sample[key] 做側錄
    def _tokenize_dynamic_piece(self, sample: dict, piece_spec: str):
        # piece_spec 是花括號內的原字串；可能以 '!' 開頭表示 non-pooling
        non_pool = piece_spec.startswith("!")
        inner = piece_spec[1:] if non_pool else piece_spec

        cache = {}

        def repl(m):
            key = m.group(1)
            raw = sample.get(key, "")
            if key not in cache:
                cache[key] = raw  # 先不 clamp，稍後統一計算 raw_len
            return cache[key]

        filled_raw = re.sub(r"%%(.*?)%%", repl, inner)

        # ---- 先計算「未截斷」的 raw_len ----
        tok_raw = self.tokenizer(
            filled_raw, truncation=False, padding=False, add_special_tokens=False
        )
        raw_ids = tok_raw["input_ids"]
        raw_len = len(raw_ids)

        # ---- 再做 piece 級別的 clamp，得到 post_clamp_len ----
        if self.piece_max_tokens is not None and raw_len > self.piece_max_tokens:
            # 保留前段
            clamped_ids = raw_ids[: self.piece_max_tokens]
            # 保留後段
            # clamped_ids = raw_ids[self.piece_max_tokens * -1 :]
            # 保留中間
            # clamped_ids = raw_ids[
            #     (raw_len - self.piece_max_tokens)
            #     // 2 : (raw_len + self.piece_max_tokens)
            #     // 2
            # ]
        else:
            clamped_ids = raw_ids
        post_len = len(clamped_ids)

        input_ids = torch.tensor(clamped_ids, dtype=torch.long)
        L = input_ids.shape[0]
        out: Dict[str, Union[torch.Tensor, int]] = {
            "input_ids": input_ids,
            "attention_mask": torch.ones(L, dtype=torch.long),
            "embed_mask": (
                torch.zeros(L, dtype=torch.long)
                if non_pool
                else torch.ones(L, dtype=torch.long)
            ),
            "_raw_len": raw_len,
            "_post_len": post_len,
        }
        return out

    def _tokenize_static_piece(self, ids: List[int]) -> Dict[str, torch.Tensor]:
        L = len(ids)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(L, dtype=torch.long),
            "embed_mask": torch.zeros(L, dtype=torch.long),  # static -> never pooled
            "_raw_len": L,
            "_post_len": L,
        }

    def _tokenize_from_pieces(
        self, sample: Dict[str, str], pieces: List[Union[str, List[int]]]
    ) -> Dict[str, torch.Tensor]:
        outs: List[Dict[str, Union[torch.Tensor, int]]] = []
        raw_total = 0
        post_total = 0
        for p in pieces:
            if isinstance(p, str):
                o = self._tokenize_dynamic_piece(sample, p)
            else:
                o = self._tokenize_static_piece(p)
            outs.append(o)  # type: ignore[arg-type]
            raw_total += int(o.get("_raw_len", 0))  # type: ignore[union-attr]
            post_total += int(o.get("_post_len", 0))  # type: ignore[union-attr]

        # concat along sequence
        def cat(key: str) -> torch.Tensor:
            return (
                torch.cat([o[key] for o in outs], dim=0)  # type: ignore[index]
                if outs
                else torch.empty(0, dtype=torch.long)
            )

        batch_row: Dict[str, torch.Tensor] = {
            "input_ids": cat("input_ids"),
            "attention_mask": cat("attention_mask"),
            "embed_mask": cat("embed_mask"),
        }

        if self._collect_len_stats:
            # final_len 之後在 batching 截斷/補齊後計算
            self._row_stats.append(
                {"pre_clamp_len": raw_total, "post_clamp_len": post_total}
            )

        return batch_row

    def tokenize(
        self,
        items: Sequence[Union[Tuple[str, str], Tuple[str, Dict[str, str]]]],
    ) -> Dict[str, torch.Tensor]:
        """
        items: sequence of (template_key, sample) where sample is either str or dict.
               If sample is str, it will be mapped to {'x': sample}.
        """
        rows: List[Dict[str, torch.Tensor]] = []
        for key, sample in items:
            if key not in self.template_pieces:
                raise KeyError(f"Unknown template key '{key}'")
            if isinstance(sample, str):
                sample = {"x": sample}
            row = self._tokenize_from_pieces(sample, self.template_pieces[key])
            rows.append(row)

        # pad to batch
        max_len = max((r["input_ids"].shape[0] for r in rows), default=0)
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)

        if self.pad_to_multiple_of and max_len % self.pad_to_multiple_of != 0:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of
            ) * self.pad_to_multiple_of

        def pad1d(x: torch.Tensor, pad_id: int, length: int) -> torch.Tensor:
            L = x.shape[0]
            if L >= length:
                return x[:length]
            out = x.new_full((length,), pad_id)
            out[:L] = x
            return out

        pad_id = self.tokenizer.pad_token_id
        batch = {
            "input_ids": torch.stack(
                [pad1d(r["input_ids"], pad_id, max_len) for r in rows], dim=0
            ),
            "attention_mask": torch.stack(
                [pad1d(r["attention_mask"], 0, max_len) for r in rows], dim=0
            ),
            "embed_mask": torch.stack(
                [pad1d(r["embed_mask"], 0, max_len) for r in rows], dim=0
            ),
        }
        return batch

    # sugar
    def __call__(self, items):
        return self.tokenize(items)

    def get_tokenizer(self):
        return self.tokenizer


# -----------------------
# EchoPooling
# -----------------------
class EchoPooling(nn.Module):
    def __init__(self, strategy: str = "mean") -> None:
        super().__init__()
        assert strategy in ("mean", "last")
        self.strategy = strategy

    def forward(self, xs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        xs expects:
          - token_embeddings: (B, L, H)
          - embed_mask: (B, L)
        Produces:
          - sentence_embedding: (B, H)
        """
        emb = xs["token_embeddings"]
        mask = xs["embed_mask"].to(dtype=emb.dtype)  # (B, L)

        if self.strategy == "mean":
            num = torch.einsum("blh,bl->bh", emb, mask)  # sum over positions
            den = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # avoid div0
            out = num / den
        else:  # last
            B, L, H = emb.shape
            idx = xs["embed_mask"] > 0
            # default 0
            last_pos = torch.zeros(B, dtype=torch.long, device=emb.device)
            for b in range(B):
                pos = torch.nonzero(idx[b], as_tuple=False).flatten()
                if len(pos) > 0:
                    last_pos[b] = pos[-1]
            out = emb[torch.arange(B, device=emb.device), last_pos]  # (B, H)

        xs["sentence_embedding"] = out
        return xs


# -----------------------
# EchoEmbeddingsModel
# -----------------------
class EchoEmbeddingsModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, xs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        inputs = {
            "input_ids": xs["input_ids"].to(self.model.device),
            "attention_mask": xs["attention_mask"].to(self.model.device),
        }
        outputs = self.model(**inputs, use_cache=False).last_hidden_state
        xs["token_embeddings"] = outputs
        return xs

    def from_pretrained(base_model: str, **kwargs) -> "EchoEmbeddingsModel":
        model = AutoModel.from_pretrained(base_model, **kwargs)
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        return EchoEmbeddingsModel(model)


# -----------------------
# High-level: EchoBatched
# -----------------------
@dataclass
class EchoBatchedConfig:
    base_model: str
    tokenizer: Optional[str] = None
    templates: Optional[Dict[str, str]] = None
    pooling: str = "mean"
    max_length: Optional[int] = 600
    piece_max_tokens: int = 256
    pad_to_multiple_of: Optional[int] = 8
    dtype: Optional[torch.dtype] = None  # e.g., torch.bfloat16/float16
    device: Optional[str] = None  # 'cuda'|'cpu'|None -> auto
    use_dataparallel: bool = True  # enable if multi-GPU available


class EchoBatched:
    def __init__(self, cfg: EchoBatchedConfig) -> None:
        self.cfg = cfg

        tok_name = cfg.tokenizer or cfg.base_model
        self.parser = EchoParser(
            tokenizer=tok_name,
            templates=cfg.templates
            or {
                "query": "{!Rewrite the following sentence: %%x%%}\n{The rewritten sentence: %%x%%}"
            },
            max_length=cfg.max_length,
            piece_max_tokens=cfg.piece_max_tokens,
            pad_to_multiple_of=cfg.pad_to_multiple_of,
        )

        self.model = EchoEmbeddingsModel.from_pretrained(cfg.base_model)
        self.pool = EchoPooling(cfg.pooling)

        # device & DDP-style
        dev = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)
        self.model = self.model.to(self.device)

        if (
            self.device.type == "cuda"
            and cfg.use_dataparallel
            and torch.cuda.device_count() > 1
        ):
            self.model = torch.nn.DataParallel(self.model)

        # dtype for autocast
        self.amp_dtype = cfg.dtype

        # perf knobs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        # ---- length stats accumulators ----
        self._len_acc_pre: List[int] = []
        self._len_acc_post: List[int] = []
        self._len_acc_final: List[int] = []

        self.parser.enable_length_stats(True)

    def _template_key(self, prompt_type: Optional[PromptType]) -> str:
        if prompt_type == PromptType.query and "query" in self.parser.templates:
            return "query"
        if prompt_type == PromptType.passage and "document" in self.parser.templates:
            return "document"
        # default fallbacks
        if "query" in self.parser.templates:
            return "query"
        return next(iter(self.parser.templates.keys()))

    @torch.no_grad()
    def encode(
        self,
        texts: Sequence[str],
        prompt_type: Optional[PromptType] = None,
        batch_size: int = 16,
        progress: bool = False,
        task_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        True batched encoding:
         1) Batch-tokenize with EchoParser (producing embed_mask aligned per item)
         2) Forward the whole batch (or chunks) through the backbone with AMP
         3) Pool with embed_mask to get sentence embeddings
        """
        key = self._template_key(prompt_type)

        # chunking for memory friendliness
        out_chunks: List[np.ndarray] = []
        rng = range(0, len(texts), batch_size)
        it = (
            rng
            if not progress
            else __import__("tqdm").tqdm(rng, desc=f"Encoding({task_name})")
        )
        for i in it:
            pairs = [(key, t) for t in texts[i : i + batch_size]]
            tokens = self.parser(pairs)

            # --- collect pre/post clamp lengths from parser (per-row) ---
            for s in self.parser.pop_length_stats():
                self._len_acc_pre.append(int(s["pre_clamp_len"]))
                self._len_acc_post.append(int(s["post_clamp_len"]))

            # move to device for the forward pass only
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            # final length after batching (truncate/pad)
            self._len_acc_final.extend(
                tokens["attention_mask"].sum(dim=1).cpu().tolist()
            )

            # autocast
            if self.device.type == "cuda":
                autocast_ctx = torch.amp.autocast(
                    "cuda", dtype=self.amp_dtype
                )  # torch 2.0+
            else:

                class _NoCtx:
                    def __enter__(self, *a):
                        return None

                    def __exit__(self, *a):
                        return False

                autocast_ctx = _NoCtx()

            with autocast_ctx:
                xs = self.model(tokens)  # adds token_embeddings

            # pooling needs embed_mask
            xs["embed_mask"] = tokens["embed_mask"]
            xs = self.pool(xs)

            emb = xs["sentence_embedding"].detach().float().cpu().numpy()
            out_chunks.append(emb)

        def _summ(name: str, arr_list: List[int]) -> str:
            arr = np.asarray(arr_list, dtype=np.int64)
            if arr.size == 0:
                return f"{name}: n=0"
            p50, p90, p95, p99 = np.percentile(arr, [50, 90, 95, 99]).tolist()
            return (
                f"{name}: n={arr.size} mean={arr.mean():.1f} "
                f"median={p50:.0f} p90={p90:.0f} p95={p95:.0f} p99={p99:.0f} "
                f">128={(arr>128).mean()*100:.1f}% >256={(arr>256).mean()*100:.1f}% >512={(arr>512).mean()*100:.1f}%"
            )

        print("[LEN] ", _summ("pre_clamp", self._len_acc_pre))
        print("[LEN] ", _summ("post_clamp", self._len_acc_post))
        print("[LEN] ", _summ("final", self._len_acc_final))

        # Clear accumulators for the next task
        self._len_acc_pre.clear()
        self._len_acc_post.clear()
        self._len_acc_final.clear()

        return np.vstack(out_chunks)


# -----------------------
# Optional: MTEB adapter
# -----------------------
class EchoModel:
    """
    Minimal adapter so you can do:
        model = EchoModel(path_to_model, templates, max_length=300, pooling_strategy="mean")
        embs = model.encode(texts, prompt_type=PromptType.query, batch_size=64)
    """

    def __init__(
        self,
        path_to_model: str,
        templates: Dict[str, str],
        max_length: int = 600,
        piece_max_tokens: int = 256,
        pooling_strategy: str = "mean",
        pad_to_multiple_of: Optional[int] = 8,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        cfg = EchoBatchedConfig(
            base_model=path_to_model,
            tokenizer=path_to_model,
            templates=templates,
            pooling=pooling_strategy,
            max_length=max_length,
            piece_max_tokens=piece_max_tokens,
            pad_to_multiple_of=pad_to_multiple_of,
            dtype=dtype,
        )
        self.engine = EchoBatched(cfg)

    def encode(
        self,
        sentences: Sequence[str],
        prompt_type: Optional[PromptType] = None,
        task_name: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        return self.engine.encode(
            sentences,
            prompt_type=prompt_type,
            batch_size=kwargs.get("batch_size", 32),
            progress=kwargs.get("show_progress_bar", True),
            task_name=task_name,
        )
