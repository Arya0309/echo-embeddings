#!/usr/bin/env python3
# echo_fig2_eval.py
# Usage examples:
#   python echo_fig2_eval.py --json /path/echo_toy_triplets.json --model mistralai/Mistral-7B-Instruct-v0.1 --mode classical --pool mean
#   python echo_fig2_eval.py --json /path/echo_toy_triplets.json --model mistralai/Mistral-7B-Instruct-v0.1 --mode echo --pool mean
#   python echo_fig2_eval.py --json /path/echo_toy_triplets.json --mode echo --pool last --max-length 300

import argparse
import json
from typing import Dict, Any, Tuple, List

import torch
from echo_embeddings import EchoEmbeddingsMistral, EchoPooling, EchoParser


# ---------- Templates (Appendix C-aligned) ----------
ECHO_TEMPLATES = {
    # 第一段 {!%%text%%} 僅作上下文；第二段 {%%text%%} 會被 pooling
    "query": "<s>Rewrite the following sentence: {!%%text%%}\nThe rewritten sentence: {%%text%%}{</s>}",
    "document": "<s>Rewrite the following sentence: {!%%text%%}\nThe rewritten sentence: {%%text%%}{</s>}",
}

CLASSICAL_TEMPLATES = {
    # 單份 S，被 pooling 的 span
    "query": "<s>Write a sentence: {%%text%%}{</s>}",
    "document": "<s>Write a sentence: {%%text%%}{</s>}",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Reproduce Echo vs Classical on toy triplets (Fig. 2-style)."
    )
    ap.add_argument(
        "--json",
        type=str,
        default="echo_toy_examples.json",
        help="Path to toy triplets JSON (with fields q, s_pos/s_neg or s+/s-)",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.1",
        help="HF model id or local path",
    )
    ap.add_argument(
        "--mode",
        type=str,
        choices=["echo", "classical"],
        default="echo",
        help="Use echo (two copies) or classical (single copy)",
    )
    ap.add_argument(
        "--pool",
        type=str,
        choices=["mean", "last"],
        default="mean",
        help="Pooling strategy",
    )
    ap.add_argument(
        "--max-length", type=int, default=300, help="Parser max sequence length"
    )
    ap.add_argument(
        "--prompt",
        type=str,
        default="Retrieve passages that answer the question",
        help="Instruction string (kept constant across modes)",
    )
    ap.add_argument("--quiet", action="store_true", help="Less per-example logging")
    return ap.parse_args()


def pick_templates(mode: str) -> Dict[str, str]:
    return ECHO_TEMPLATES if mode == "echo" else CLASSICAL_TEMPLATES


def build_model_and_tools(
    model_path: str, templates: Dict[str, str], max_length: int, pool: str
):
    model = EchoEmbeddingsMistral.from_pretrained(model_path).eval()
    parser = EchoParser(model_path, templates, max_length=max_length)
    pooling = EchoPooling(strategy=pool)
    return model, parser, pooling


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
    return (torch.dot(x, y) / (torch.norm(x) * torch.norm(y))).item()


def embed_sentence(
    model, parser, pooling, tag: str, text: str, prompt: str = None
) -> torch.Tensor:
    # `tag` in {"query","document"}
    variables = (
        [{"text": text}] if prompt is None else [{"prompt": prompt, "text": text}]
    )
    tagged = [(tag, v) for v in variables]
    with torch.no_grad():
        out = model(parser(tagged))
        pooled = pooling(out)["sentence_embedding"][0]  # take first item of batch
    return pooled


def read_triplet(example: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """Robustly read q, s_pos, s_neg, structure (accept s+ / s- aliases)."""
    q = example["q"]
    s_pos = example.get("s_pos", example.get("s+"))
    s_neg = example.get("s_neg", example.get("s-"))
    if s_pos is None or s_neg is None:
        raise KeyError("Example must contain s_pos/s_neg (or s+/s-).")
    structure = example.get("structure", "ALL")
    return q, s_pos, s_neg, structure


def main():
    args = parse_args()

    templates = pick_templates(args.mode)
    model, parser, pooling = build_model_and_tools(
        args.model, templates, args.max_length, args.pool
    )

    with open(args.json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # 支援兩種格式：{"data":[...]} 或直接是 list
    examples: List[Dict[str, Any]] = (
        payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    )

    total = corr = 0
    total_s1 = corr_s1 = 0
    total_s2 = corr_s2 = 0

    # （選配）收集 S2 分佈可作 Fig. 2C 直方圖
    s2_pos_scores, s2_neg_scores = [], []

    for ex in examples:
        q, s_pos, s_neg, struct = read_triplet(ex)

        q_emb = embed_sentence(model, parser, pooling, "query", q, prompt=args.prompt)
        sp_emb = embed_sentence(model, parser, pooling, "document", s_pos)
        sn_emb = embed_sentence(model, parser, pooling, "document", s_neg)

        sim_pos = cosine_similarity(q_emb, sp_emb)
        sim_neg = cosine_similarity(q_emb, sn_emb)

        is_correct = sim_pos > sim_neg
        corr += int(is_correct)
        total += 1

        if struct == "S1":
            total_s1 += 1
            corr_s1 += int(is_correct)
        elif struct == "S2":
            total_s2 += 1
            corr_s2 += int(is_correct)
            s2_pos_scores.append(sim_pos)
            s2_neg_scores.append(sim_neg)

        if not args.quiet:
            status = "Correct" if is_correct else "Wrong  "
            print(
                f"{status}: {sim_pos:.4f} vs {sim_neg:.4f} | struct={struct} | q={q[:80]}"
            )

    def safe_acc(c, t):
        return (c / t) if t > 0 else float("nan")

    print("\n=== Results ===")
    print(f"Mode={args.mode} | Pool={args.pool} | Model={args.model}")
    print(f"ALL: acc={safe_acc(corr, total):.4f}  (n={total})")
    print(f"S1 : acc={safe_acc(corr_s1, total_s1):.4f} (n={total_s1})")
    print(f"S2 : acc={safe_acc(corr_s2, total_s2):.4f} (n={total_s2})")

    # 若你想輸出 S2 分佈做 Fig.2C，可另行存檔
    # import numpy as np, json
    # json.dump({"s2_pos": s2_pos_scores, "s2_neg": s2_neg_scores}, open("s2_scores.json","w"))


if __name__ == "__main__":
    main()
