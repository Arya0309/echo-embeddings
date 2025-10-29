import logging, sys

# 把 root logger 開到 INFO，強制覆蓋既有設定（在 notebook / 多次執行時很重要）
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,  # Py3.8+，確保真的套用
)

import random
import numpy as np
import torch

# 設定全域隨機種子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# main.py
import mteb
from echo_for_mteb import EchoModel
from listing_mteb import by_type


def run(
    templates,
    output_folder,
    tasks_list=None,
    piece_max_tokens=256,
    max_length=600,
    batch_size=32,
):
    model = EchoModel(
        "mistralai/Mistral-7B-Instruct-v0.1",
        templates,
        pooling_strategy="mean",
        piece_max_tokens=piece_max_tokens,
        max_length=max_length,
    )

    tasks = mteb.get_tasks(
        tasks=tasks_list, languages=["eng"], exclusive_language_filter=True
    )

    evaluation = mteb.MTEB(tasks=tasks)
    evaluation.run(
        model,
        encode_kwargs={
            "batch_size": batch_size,
            "show_progress_bar": True,
        },
        output_folder=output_folder,
        verbosity=1,
    )


# templates_echo = {
#     "query": "<s>Rewrite the following sentence: {!%%x%%}\nThe rewritten sentence: {%%x%%}</s>",
#     "document": "<s>Rewrite the following sentence: {!%%x%%}\nThe rewritten sentence: {%%x%%}</s>",
# }

templates_echo = {
    "query": "<s>Rewrite the following paragraph: {!%%x%%}\nThe rewritten paragraph: {%%x%%}</s>",
    "document": "<s>Rewrite the following paragraph: {!%%x%%}\nThe rewritten paragraph: {%%x%%}</s>",
}

# templates_classical = {
#     "query": "<s>Rewrite the following sentence: {%%x%%}</s>",
#     "document": "<s>Rewrite the following sentence: {%%x%%}</s>",
# }

templates_classical = {
    "query": "<s>Rewrite the following paragraph: {%%x%%}</s>",
    "document": "<s>Rewrite the following paragraph: {%%x%%}</s>",
}


if __name__ == "__main__":
    tasks_list = (
        by_type["Reranking"]
        + by_type["STS"]
        + by_type["Clustering"]
        + by_type["PairClassification"]
        + by_type["Summarization"]
        + "ArguAna"
    )

    run(
        templates_echo,
        "mteb_results/echo_mistral/ALL/mid_clamp_256",
        tasks_list=tasks_list,
        piece_max_tokens=256,
        max_length=600,
        batch_size=32,
    )
    # run(
    #     templates_echo,
    #     "mteb_results/echo_mistral/ALL/mid_clamp_128",
    #     tasks_list=tasks_list,
    #     piece_max_tokens=128,
    #     max_length=300,
    #     batch_size=32,
    # )
    # run(
    #     templates_echo,
    #     "mteb_results/echo_mistral/ALL/mid_clamp",
    #     tasks_list=tasks_list,
    #     piece_max_tokens=512,
    #     max_length=1100,
    #     batch_size=16,
    # )
    # run(templates_classical, "mteb_results/mistral", tasks_list=tasks_list)
