#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, csv, glob, argparse, threading
from typing import Any, Dict, List, Optional, Sequence

from mteb import MTEB, get_tasks

# 可選載入：用 by_type 取得資料集清單；若沒有這個檔，改用 --datasets
try:
    from listing_mteb import by_type
except Exception:
    by_type = None

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np


# =========================
# Length Probe (wrap model)
# =========================
class _LenCSV:
    """Thread-safe CSV writer per task_name."""

    def __init__(self, out_dir: str, task_name: str):
        os.makedirs(out_dir, exist_ok=True)
        safe = "".join(
            ch if ch.isalnum() or ch in "-_." else "_"
            for ch in (task_name or "UNKNOWN_TASK")
        )
        self.path = os.path.join(out_dir, f"{safe}_len.csv")
        self._fh = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._fh)
        self._w.writerow(["prompt_type", "raw_len", "trunc_len", "text"])
        self._lock = threading.Lock()

    def write_rows(self, rows: List[List[Any]]):
        with self._lock:
            self._w.writerows(rows)

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


class LenProbeWrapper:
    """
    Wrap a SentenceTransformer model and log token length for the EXACT inputs that MTEB passes.
    - raw_len: tokenizer(..., add_special_tokens=False, truncation=False)
    - trunc_len: tokenizer(..., add_special_tokens=True, truncation=True, max_length=model_max_length)
    """

    def __init__(
        self, base_st_model: SentenceTransformer, tokenizer_id: str, out_dir: str
    ):
        self.base = base_st_model
        self.tok = AutoTokenizer.from_pretrained(tokenizer_id)
        if getattr(self.tok, "pad_token", None) is None:
            try:
                self.tok.pad_token = self.tok.eos_token or self.tok.unk_token or "<pad>"
            except Exception:
                self.tok.add_special_tokens({"pad_token": "<pad>"})
        self.out_dir = out_dir
        self._writers: Dict[str, _LenCSV] = {}

    def _writer(self, task_name: Optional[str]) -> _LenCSV:
        key = task_name or "UNKNOWN_TASK"
        if key not in self._writers:
            self._writers[key] = _LenCSV(self.out_dir, key)
        return self._writers[key]

    # 讓 MTEB 呼叫：簽名支援 prompt_type / task_name，但我們只用來記錄
    def encode(
        self,
        sentences: Sequence[str],
        prompt_type: Optional[Any] = None,
        task_name: Optional[str] = None,
        **kwargs,
    ):
        texts = list(sentences)

        # raw (no specials, no trunc)
        enc_raw = self.tok(
            texts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        raw_lens = [len(ids) for ids in enc_raw["input_ids"]]

        # trunc (with specials, trunc to model_max_length)
        enc_trunc = self.tok(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=self.tok.model_max_length,
            padding=False,
            return_attention_mask=False,
        )
        trunc_lens = [len(ids) for ids in enc_trunc["input_ids"]]

        # 寫入 CSV
        writer = self._writer(task_name)
        pt = (
            getattr(prompt_type, "value", None)
            if hasattr(prompt_type, "value")
            else prompt_type
        )
        rows = [[pt or "", r, t, txt] for r, t, txt in zip(raw_lens, trunc_lens, texts)]
        writer.write_rows(rows)

        # 交給真正模型計算 embedding（維持原行為）
        return self.base.encode(texts, **kwargs)

    # 其餘屬性轉發
    def __getattr__(self, name):
        return getattr(self.base, name)


def wrap_with_len_probe_st(
    st_model_id: str, tok_for_len_id: str, out_dir: str
) -> LenProbeWrapper:
    base = SentenceTransformer(st_model_id)
    return LenProbeWrapper(base, tokenizer_id=tok_for_len_id, out_dir=out_dir)


# ===================
# Tasks list helpers
# ===================
def tasks_from_bytype(categories: List[str]) -> List[str]:
    if by_type is None:
        raise RuntimeError(
            "by_type not available; either provide listing_mteb.by_type or use --datasets"
        )
    seen, out = set(), []
    for cat in categories:
        for t in by_type.get(cat, []):
            if t not in seen:
                out.append(t)
                seen.add(t)
    return out


# =========================
# Summarize CSVs to a table
# =========================
def summarize_probe_dir(in_dir: str, out_csv: str):
    rows_out = []
    for path in glob.glob(os.path.join(in_dir, "*_len.csv")):
        ds = os.path.basename(path).replace("_len.csv", "")
        raw, trunc = [], []
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    raw.append(int(row["raw_len"]))
                    trunc.append(int(row["trunc_len"]))
                except Exception:
                    pass
        for kind, arr in [("raw", raw), ("trunc", trunc)]:
            if not arr:
                rows_out.append({"dataset": ds, "kind": kind, "n": 0})
                continue
            a = np.array(arr, dtype=np.int64)
            p50, p90, p95, p99 = np.percentile(a, [50, 90, 95, 99]).tolist()
            rows_out.append(
                {
                    "dataset": ds,
                    "kind": kind,
                    "n": int(a.size),
                    "mean": float(a.mean()),
                    "median": float(p50),
                    "p90": float(p90),
                    "p95": float(p95),
                    "p99": float(p99),
                    ">128(%)": float((a > 128).mean() * 100),
                    ">256(%)": float((a > 256).mean() * 100),
                    ">512(%)": float((a > 512).mean() * 100),
                }
            )
    if rows_out:
        cols = [
            "dataset",
            "kind",
            "n",
            "mean",
            "median",
            "p90",
            "p95",
            "p99",
            ">128(%)",
            ">256(%)",
            ">512(%)",
        ]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows_out:
                w.writerow(r)
        print(f"[OK] wrote summary: {out_csv}")
    else:
        print("[WARN] no *_len.csv files found in", in_dir)


# ==========
# CLI: run / summarize
# ==========
def cli_run(args: argparse.Namespace):
    # 1) 建立普通的 SentenceTransformer 模型 + 長度量測包裝
    model = wrap_with_len_probe_st(
        st_model_id=args.st_model,
        tok_for_len_id=args.tok_for_len,
        out_dir=args.probe_out,
    )

    # 2) 任務清單
    if args.from_bytype and (not args.datasets):
        ds_names = tasks_from_bytype(args.categories)
        print(f"[by_type] categories={args.categories} -> {len(ds_names)} datasets")
    else:
        ds_names = args.datasets or []
        print(f"[explicit] datasets -> {len(ds_names)} datasets")

    tasks = get_tasks(tasks=ds_names) if ds_names else get_tasks()

    # 3) 跑 MTEB（可選擇每任務限制 N 筆）
    evaluation = MTEB(tasks=tasks)
    evaluation.run(
        model=model,
        output_folder=args.results,
        limit=(
            args.per_task_limit
            if args.per_task_limit and args.per_task_limit > 0
            else None
        ),
        verbosity=1,
        encode_kwargs={
            "batch_size": 256,
        },
    )
    print(f"[DONE] MTEB run complete. Probe CSVs at: {args.probe_out}")


def cli_summarize(args: argparse.Namespace):
    summarize_probe_dir(args.in_dir, args.out)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="mteb_len_probe_st",
        description="Run MTEB with a SentenceTransformer model while recording input token lengths.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # run
    p_run = sub.add_parser(
        "run", help="Run MTEB while recording token lengths for each encode() call."
    )
    p_run.add_argument(
        "--st-model",
        required=True,
        help="SentenceTransformer model id, e.g., sentence-transformers/all-mpnet-base-v2",
    )
    p_run.add_argument(
        "--tok-for-len",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Tokenizer used to COUNT tokens; independent from embedding model if you want.",
    )
    p_run.add_argument(
        "--results", default="./mteb_results/st_results", help="MTEB results folder"
    )
    p_run.add_argument(
        "--probe-out", default="./len_probe_out", help="Folder to write *_len.csv files"
    )
    p_run.add_argument(
        "--from-bytype",
        action="store_true",
        help="Use listing_mteb.by_type to pick datasets",
    )
    p_run.add_argument(
        "--categories",
        nargs="+",
        default=[
            "Reranking",
            "STS",
            "Clustering",
            "PairClassification",
            "Summarization",
        ],
    )
    p_run.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Explicit dataset names (override by_type if given)",
    )
    p_run.add_argument(
        "--per-task-limit",
        type=int,
        default=0,
        help="Optional limit per task (0=full task)",
    )
    p_run.set_defaults(func=cli_run)

    # summarize
    p_sum = sub.add_parser("summarize", help="Summarize *_len.csv files into one CSV.")
    p_sum.add_argument("--in", dest="in_dir", default="./len_probe_out")
    p_sum.add_argument("--out", dest="out", default="./len_summary_from_probe.csv")
    p_sum.set_defaults(func=cli_summarize)

    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
