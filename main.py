# main.py
import mteb
from echo_for_mteb import EchoModel
from listing_mteb import by_type

templates = {
    "query": "<s>Rewrite the following sentence: {!%%x%%}\nThe rewritten sentence: {%%x%%}</s>",
    "document": "<s>Rewrite the following sentence: {!%%x%%}\nThe rewritten sentence: {%%x%%}</s>",
}

model = EchoModel(
    "mistralai/Mistral-7B-Instruct-v0.1", templates, pooling_strategy="mean"
)

tasks = mteb.get_tasks(tasks=by_type["Classification"][:6])

evaluation = mteb.MTEB(tasks=tasks)
evaluation.run(
    model,
    encode_kwargs={
        "batch_size": 16,
        "show_progress_bar": True,
    },
    output_folder="mteb_results/echo_mistral",
)
