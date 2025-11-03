#!/usr/bin/env python3
# main_toy_example.py  (Stanza-aware WEIGHTED pooling: keep words but exclude low-score tokens from pooling)
# Usage examples:
#   python main_toy_example.py --json /path/echo_toy_triplets.json --model mistralai/Mistral-7B-Instruct-v0.1 --mode classical --pool mean
#   python main_toy_example.py --json /path/echo_toy_triplets.json --model mistralai/Mistral-7B-Instruct-v0.1 --mode echo --pool mean --stanza-filter --keep-top-p 0.4
#   python main_toy_example.py --stanza-filter --score-threshold 3

import argparse
import json
from typing import Dict, Any, Tuple, List, Optional

import torch
from transformers import AutoTokenizer
from echo_embeddings import EchoEmbeddingsMistral, EchoPooling, EchoParser


# ---------- Templates (context/text split for Echo) ----------
ECHO_TEMPLATES = {
    # 第一段 {!%%context%%} 不參與池化；第二段 {%%text%%} 參與池化
    "query": "<s>Rewrite the following sentence: {!%%context%%}\nThe rewritten sentence: {%%text%%}{</s>}",
    "document": "<s>Rewrite the following sentence: {!%%context%%}\nThe rewritten sentence: {%%text%%}{</s>}",
}

CLASSICAL_TEMPLATES = {
    "query": "<s>Write a sentence: {%%text%%}{</s>}",
    "document": "<s>Write a sentence: {%%text%%}{</s>}",
}


# =========================
# Stanza-based scoring (keep words, control pooling weights)
# =========================
def _lazy_load_stanza(
    lang: str = "en", processors: str = "tokenize,pos,lemma,depparse,ner"
):
    import stanza

    try:
        nlp = stanza.Pipeline(lang, processors=processors, tokenize_no_ssplit=True)
    except Exception:
        stanza.download(lang)
        nlp = stanza.Pipeline(lang, processors=processors, tokenize_no_ssplit=True)
    return nlp


_POS_W = {"VERB": 3, "NOUN": 3, "PROPN": 3, "ADJ": 2, "ADV": 2, "NUM": 2}
_DEP_W = {
    "root": 4,
    "xcomp": 3,
    "ccomp": 3,
    "advcl": 2,
    "nsubj": 3,
    "obj": 3,
    "iobj": 2,
    "obl": 2,
    "amod": 2,
    "advmod": 2,
    "nummod": 2,
    "compound": 1,
    "appos": 1,
    "conj": 2,
    "neg": 4,
}
_LOW_INFO_DEPS = {"case", "mark", "cc", "punct"}


def stanza_token_score(upos: str, deprel: str, ner: Optional[str]) -> int:
    pos_w = _POS_W.get(upos, 0)
    dep_w = _DEP_W.get(deprel, 0)
    if deprel in _LOW_INFO_DEPS:
        dep_w = 0
    ner_w = 2 if (ner and ner != "O") else 0
    return pos_w + dep_w + ner_w


def _word_scores_from_stanza(nlp, text: str) -> List[Tuple[Tuple[int, int], int]]:
    """回傳 [(char_span, score), ...] 依原文字符範圍與打分。"""
    if not text.strip():
        return []
    doc = nlp(text)
    if not doc.sentences:
        return []
    words = doc.sentences[0].words  # word-level（多詞 token 已拆）
    # NER 在 token 層，不逐一對齊以簡化（多數為 'O'），需要可擴充
    scored = []
    for w in words:
        upos = w.upos or "_"
        deprel = w.deprel or "_"
        # 拿 character offsets：Stanza word 有 start_char/end_char
        # 若缺失，跳過（極少情況）
        if w.start_char is None or w.end_char is None:
            continue
        score = stanza_token_score(upos, deprel, None)
        scored.append(((int(w.start_char), int(w.end_char)), score))
    return scored


def _decide_binary_keep(
    scores: List[int], keep_top_p: Optional[float], score_threshold: Optional[int]
) -> List[int]:
    """把整詞分數轉成 0/1（1=參與池化）。"""
    if not scores:
        return []
    if keep_top_p is not None:
        assert 0.0 < keep_top_p <= 1.0
        sorted_scores = sorted(scores, reverse=True)
        k = max(1, int(len(scores) * keep_top_p))
        thresh = sorted_scores[k - 1]
        return [1 if s >= thresh else 0 for s in scores]
    if score_threshold is not None:
        return [1 if s >= score_threshold else 0 for s in scores]
    # 預設：Top-40%
    sorted_scores = sorted(scores, reverse=True)
    k = max(1, int(len(scores) * 0.4))
    thresh = sorted_scores[k - 1]
    return [1 if s >= thresh else 0 for s in scores]


def build_subword_weights_for_text(
    nlp,
    tokenizer: AutoTokenizer,
    text: str,
    keep_top_p: Optional[float],
    score_threshold: Optional[int],
) -> List[float]:
    """
    以 Stanza 的詞級打分，對齊到 HF 子詞，產生一個 subword 權重向量（0/1）。
    不刪字，只控制池化權重。
    """
    if nlp is None:
        # 不做任何限制：全部設 1（讓 embed_mask 決定哪些位置參與池化）
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        return [1.0] * len(enc["offset_mapping"])

    # 1) 取詞級 (char_start, char_end, score)
    span_scores = _word_scores_from_stanza(nlp, text)
    if not span_scores:
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        return [1.0] * len(enc["offset_mapping"])

    spans, scores = zip(*span_scores)  # lists
    keep_bits = _decide_binary_keep(list(scores), keep_top_p, score_threshold)

    # 2) 對齊到子詞 offsets
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    sw_offsets = enc["offset_mapping"]  # [(s,e), ...]
    sw_weights = []
    for sw_s, sw_e in sw_offsets:
        if sw_e <= sw_s:  # 空白或特殊情況
            sw_weights.append(0.0)
            continue
        # 找到涵蓋此 subword 的詞（以有交集判定）
        w = 0.0
        for i, (ws, we) in enumerate(spans):
            if not (sw_e <= ws or we <= sw_s):  # 交集
                if keep_bits[i] == 1:
                    w = 1.0
                    break
        sw_weights.append(w)
    # 如果全 0，至少保留一個 subword 以避免全空
    if all(w == 0.0 for w in sw_weights) and len(sw_weights) > 0:
        sw_weights[0] = 1.0
    return sw_weights


# =========================
# 原始腳本 + 權重池化
# =========================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Reproduce Echo vs Classical on toy triplets with optional Stanza-based weighted pooling (keep text; zero-out unimportant tokens in pooling)."
    )
    ap.add_argument(
        "--json",
        type=str,
        default="./data/echo_toy_examples.json",
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
        help="Pooling strategy (base, before weighting)",
    )
    ap.add_argument(
        "--max-length", type=int, default=512, help="Parser max sequence length"
    )
    ap.add_argument(
        "--prompt",
        type=str,
        default="Retrieve passages that answer the question",
        help="Instruction string (kept constant across modes)",
    )
    ap.add_argument("--quiet", action="store_true", help="Less per-example logging")

    # NEW: Stanza weighting options (0/1)
    ap.add_argument(
        "--stanza-filter",
        action="store_true",
        help="Enable Stanza-based token weighting for the pooled span (keep text; control pooling).",
    )
    ap.add_argument(
        "--keep-top-p",
        type=float,
        default=None,
        help="Keep the top-p proportion of tokens by score (e.g., 0.4). Takes precedence over score-threshold.",
    )
    ap.add_argument(
        "--score-threshold",
        type=int,
        default=None,
        help="Keep tokens whose score >= threshold (e.g., 3).",
    )
    ap.add_argument(
        "--stanza-lang",
        type=str,
        default="en",
        help="Language code for Stanza (default: en).",
    )
    return ap.parse_args()


def pick_templates(mode: str) -> Dict[str, str]:
    return ECHO_TEMPLATES if mode == "echo" else CLASSICAL_TEMPLATES


def build_model_and_tools(
    model_path: str,
    templates: Dict[str, str],
    max_length: int,
    pool: str,
    use_stanza: bool,
    stanza_lang: str,
):
    model = EchoEmbeddingsMistral.from_pretrained(model_path).eval().to("cuda")
    parser = EchoParser(model_path, templates, max_length=max_length)
    pooling = EchoPooling(strategy=pool)
    nlp = _lazy_load_stanza(stanza_lang) if use_stanza else None
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, add_bos_token=False, add_eos_token=False
    )
    if tokenizer.padding_side != "right":
        tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        try:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "<unk>"
        except Exception:
            pass
    return model, parser, pooling, nlp, tokenizer


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
    return (torch.dot(x, y) / (torch.norm(x) * torch.norm(y))).item()


def _context_and_pool_text(mode: str, raw_text: str) -> Tuple[Optional[str], str]:
    """Echo: (context=raw_text, pooled=raw_text). Classical: (None, raw_text)."""
    if mode == "echo":
        return raw_text, raw_text
    return None, raw_text


def _get_sequence_hidden(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    取得序列隱層 (1, L, H)：
    - 優先使用 echo 封裝常見的 "token_embeddings"
    - 退回到 HF 標準的 "last_hidden_state"
    - 或 "hidden_states" 的最後一層
    """
    if "token_embeddings" in out:
        return out["token_embeddings"]
    if "last_hidden_state" in out:
        return out["last_hidden_state"]
    if (
        "hidden_states" in out
        and isinstance(out["hidden_states"], (list, tuple))
        and len(out["hidden_states"]) > 0
    ):
        return out["hidden_states"][-1]
    # 把可用鍵名列出來，方便除錯
    avail = ", ".join(out.keys())
    raise KeyError(f"No sequence hidden found. Available keys: {avail}")


def _weighted_pool_from_out(
    out: Dict[str, torch.Tensor],
    subword_weights: List[float],
) -> torch.Tensor:
    """
    基於模型 forward 的輸出做 weighted mean：
    - seq_hidden: (1, L, H)
    - embed_mask: (1, L) → 1 表示屬於 {%%text%%} 的 token（可池化候選）
    - subword_weights：長度 ≈ Mask 中 1 的數量（若不等，取最小長度對齊）
    """
    seq_hidden = _get_sequence_hidden(out)  # (1, L, H)
    hidden = seq_hidden[0]  # (L, H)
    embed_mask = out["embed_mask"][0].to(hidden.dtype)  # (L,)

    seq_len = hidden.size(0)
    pool_idx = (
        (embed_mask > 0.5).nonzero(as_tuple=False).view(-1)
    )  # positions in pooled span

    weight_vec = torch.zeros(seq_len, dtype=hidden.dtype, device=hidden.device)
    n_assign = min(len(subword_weights), pool_idx.numel())
    if n_assign > 0:
        weight_vals = torch.tensor(
            subword_weights[:n_assign], dtype=hidden.dtype, device=hidden.device
        )
        weight_vec[pool_idx[:n_assign]] = weight_vals

    # 全 0 時退回原始 embed_mask（避免除以 0）
    if float(weight_vec.sum().item()) == 0.0:
        weight_vec = embed_mask

    num = (hidden * weight_vec.unsqueeze(-1)).sum(dim=0)  # (H,)
    denom = weight_vec.sum().clamp(min=1e-9)
    return num / denom


def embed_sentence(
    model,
    parser,
    pooling,
    tokenizer: AutoTokenizer,
    nlp,
    tag: str,
    text: str,
    prompt: str,
    mode: str,
    keep_top_p: Optional[float],
    score_threshold: Optional[int],
    use_stanza: bool,
) -> torch.Tensor:
    # 準備模板變數：Echo 有 context+text；Classical 只有 text
    context_text, pooled_text = _context_and_pool_text(mode, text)
    if mode == "echo":
        variables = [{"context": context_text, "text": pooled_text, "prompt": prompt}]
    else:
        variables = [{"text": pooled_text, "prompt": prompt}]

    tagged = [(tag, v) for v in variables]
    with torch.no_grad():
        out = model(parser(tagged))
        if use_stanza:
            # 對「會被池化的那段」建立子詞權重（0/1）
            subw = build_subword_weights_for_text(
                nlp, tokenizer, pooled_text, keep_top_p, score_threshold
            )
            pooled_vec = _weighted_pool_from_out(out, subw)
        else:
            # 原生池化
            pooled_vec = (
                EchoPooling(strategy="mean")(out)["sentence_embedding"][0]
                if pooling is None
                else pooling(out)["sentence_embedding"][0]
            )
    return pooled_vec


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
    model, parser, pooling, nlp, tokenizer = build_model_and_tools(
        args.model,
        templates,
        args.max_length,
        args.pool,
        use_stanza=args.stanza_filter,
        stanza_lang=args.stanza_lang,
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

    outputs = []

    for ex in examples:
        q, s_pos, s_neg, struct = read_triplet(ex)

        q_emb = embed_sentence(
            model,
            parser,
            pooling,
            tokenizer,
            nlp,
            "query",
            q,
            args.prompt,
            args.mode,
            args.keep_top_p,
            args.score_threshold,
            args.stanza_filter,
        )
        sp_emb = embed_sentence(
            model,
            parser,
            pooling,
            tokenizer,
            nlp,
            "document",
            s_pos,
            args.prompt,
            args.mode,
            args.keep_top_p,
            args.score_threshold,
            args.stanza_filter,
        )
        sn_emb = embed_sentence(
            model,
            parser,
            pooling,
            tokenizer,
            nlp,
            "document",
            s_neg,
            args.prompt,
            args.mode,
            args.keep_top_p,
            args.score_threshold,
            args.stanza_filter,
        )

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

        if not args.quiet:
            status = "Correct" if is_correct else "Wrong  "
            print(
                f"{status}: {sim_pos:.4f} vs {sim_neg:.4f} | struct={struct} | q={q[:80]}"
            )

        outputs.append(
            {
                "q": q,
                "s_pos": s_pos,
                "s_neg": s_neg,
                "struct": struct,
                "sim_pos": f"{sim_pos:.4f}",
                "sim_neg": f"{sim_neg:.4f}",
                "correct": is_correct,
                "mode": args.mode,
                "pool": args.pool,
                "stanza_filter": bool(args.stanza_filter),
                "keep_top_p": args.keep_top_p,
                "score_threshold": args.score_threshold,
            }
        )

    def safe_acc(c, t):
        return (c / t) if t > 0 else float("nan")

    print("\n=== Results ===")
    print(
        f"Mode={args.mode} | Pool={args.pool} | Model={args.model} | StanzaFilter={bool(args.stanza_filter)} | top-p={args.keep_top_p} | thr={args.score_threshold}"
    )
    print(f"ALL: acc={safe_acc(corr, total):.4f}  (n={total})")
    print(f"S1 : acc={safe_acc(corr_s1, total_s1):.4f} (n={total_s1})")
    print(f"S2 : acc={safe_acc(corr_s2, total_s2):.4f} (n={total_s2})")

    # Save results
    with open(f"results_{args.mode}_{args.pool}.json", "w", encoding="utf-8") as fout:
        json.dump(outputs, fout, ensure_ascii=False, indent=2)
    with open(f"summary_{args.mode}_{args.pool}.csv", "w", encoding="utf-8") as fout:
        fout.write("mode,pool,model,stanza_filter,top_p,threshold,subset,accuracy,n\n")
        fout.write(
            f"{args.mode},{args.pool},{args.model},{bool(args.stanza_filter)},{args.keep_top_p},{args.score_threshold},ALL,{safe_acc(corr, total):.4f},{total}\n"
        )
        fout.write(
            f"{args.mode},{args.pool},{args.model},{bool(args.stanza_filter)},{args.keep_top_p},{args.score_threshold},S1,{safe_acc(corr_s1, total_s1):.4f},{total_s1}\n"
        )
        fout.write(
            f"{args.mode},{args.pool},{args.model},{bool(args.stanza_filter)},{args.keep_top_p},{args.score_threshold},S2,{safe_acc(corr_s2, total_s2):.4f},{total_s2}\n"
        )


if __name__ == "__main__":
    main()
