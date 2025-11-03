import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(
    device
)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id
if getattr(model, "generation_config", None) is not None:
    model.generation_config.pad_token_id = tokenizer.pad_token_id

if tokenizer.pad_token_id >= model.get_input_embeddings().weight.size(0):
    model.resize_token_embeddings(len(tokenizer))

q = "She loves to travel in summer, especially to cold destinations, avoiding hot and crowded places"
s_p = "In summer, she adores traveling, specifically to chilly locations, steering clear of warm, populous areas"
s_n = (
    "She loves to travel in summer, but prefers to visit hot and bustling tourist spots"
)
template_echo = "<s>Rewrite the following sentence: {query}. The rewritten sentence:"

# template_k = '<s>This sentence: "{query}" means in a short phrase:'

template_k = 'Extract the keywords from this sentence: "{query}". The keywords are:'

template_EOL = 'This sentence: "{query}" means in one word:'


def rewrite_mean_embed(template: str, sentence: str, max_new_tokens: int = 256):
    # 1) 組 prompt 並生成 rewritten
    prompt = template.format(query=sentence)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    generated_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    # if "Extract" in template:
    #     # keyword extraction 特例處理
    #     rewritten = generated_text.split("The keywords:")[-1].strip()
    # else:
    #     rewritten = generated_text.split("The rewritten sentence:")[-1].strip()

    rewritten = generated_text.split(template)[-1].strip()

    if not rewritten:
        rewritten = generated_text[len(prompt) :].strip()

    # 2) 重新 tokenize 全序列（prompt + rewritten）
    full_text = prompt + " " + rewritten
    toks = tokenizer(full_text, return_tensors="pt")
    input_ids = toks.input_ids.to(device)  # (1, L)
    attention_mask = toks.attention_mask.to(device)  # (1, L) 只標註 pad=0，其餘=1

    # 3) 準備 embed_mask：只有 rewritten 範圍 = 1，其餘 = 0
    prefix_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    rewritten_ids = tokenizer(rewritten, return_tensors="pt").input_ids[0]
    start = len(prefix_ids)
    end = start + len(rewritten_ids)

    embed_mask = torch.zeros_like(input_ids)  # (1, L)
    embed_mask[0, start:end] = 1

    # 4) 前向一次，取最後一層 hidden states
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last = out.hidden_states[-1]  # (1, L, D)

    # 5) 論文式 pooling：對應 embed_mask=1 的 token 做 mean
    #   （這就等價於你 slicing 再 mean 的版本）
    denom = embed_mask.sum(dim=1, keepdim=True).clamp_min(1)  # 防 NaN
    sent_emb = (last * embed_mask.unsqueeze(-1)).sum(dim=1) / denom
    sent_emb = sent_emb[0]  # (D,)
    sent_emb = sent_emb / sent_emb.norm()  # 可選：L2 normalize

    return sent_emb, rewritten


if __name__ == "__main__":
    template = template_k
    # emb_q, rewritten_q = rewrite_mean_embed(template, q)
    # emb_p, rewritten_p = rewrite_mean_embed(template, s_p)
    # emb_n, rewritten_n = rewrite_mean_embed(template, s_n)

    # sim_pos = torch.cosine_similarity(emb_q.unsqueeze(0), emb_p.unsqueeze(0)).item()
    # sim_neg = torch.cosine_similarity(emb_q.unsqueeze(0), emb_n.unsqueeze(0)).item()

    # print(f"Original Query: {q}")
    # print(f"Rewritten Query: {rewritten_q}\n")
    # print(f"Positive Sentence: {s_p}")
    # print(f"Rewritten Positive: {rewritten_p}\n")
    # print(f"Negative Sentence: {s_n}")
    # print(f"Rewritten Negative: {rewritten_n}\n")
    # print(f"Cosine Similarity with Positive: {sim_pos:.4f}")
    # print(f"Cosine Similarity with Negative: {sim_neg:.4f}")

    import json

    result_list = []

    data_path = "/home/S113062628/project/echo-embeddings/data/echo_toy_examples.json"
    with open(data_path, "r") as f:
        data = json.load(f)
        data = data["data"]

    for idx, item in enumerate(data):
        emb_q, rewritten_q = rewrite_mean_embed(template, item["q"])
        emb_p, rewritten_p = rewrite_mean_embed(template, item["s+"])
        emb_n, rewritten_n = rewrite_mean_embed(template, item["s-"])

        sim_pos = torch.cosine_similarity(emb_q.unsqueeze(0), emb_p.unsqueeze(0)).item()
        sim_neg = torch.cosine_similarity(emb_q.unsqueeze(0), emb_n.unsqueeze(0)).item()

        result_list.append(
            {
                "q": item["q"],
                "rewritten_q": rewritten_q,
                "s+": item["s+"],
                "rewritten_p": rewritten_p,
                "s-": item["s-"],
                "rewritten_n": rewritten_n,
                "sim_pos": sim_pos,
                "sim_neg": sim_neg,
                "correct": sim_pos > sim_neg,
            }
        )

        print(f"Example {idx+1}:")
        print(f"Original Query: {item['q']}")
        print(f"Rewritten Query: {rewritten_q}\n")
        print(f"Positive Sentence: {item['s+']}")
        print(f"Rewritten Positive: {rewritten_p}\n")
        print(f"Negative Sentence: {item['s-']}")
        print(f"Rewritten Negative: {rewritten_n}\n")
        print(f"Cosine Similarity with Positive: {sim_pos:.4f}")
        print(f"Cosine Similarity with Negative: {sim_neg:.4f}")
        print("-" * 50)

    print(
        f"Accuracy: {sum([1 if r['correct'] else 0 for r in result_list]) / len(result_list):.4f}"
    )
