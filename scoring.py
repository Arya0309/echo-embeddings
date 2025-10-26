import os
import json
import pandas as pd
import argparse
from listing_mteb import by_type

parser = argparse.ArgumentParser(description="Process MTEB results")
parser.add_argument("--dir_path", type=str, default="/home/S113062628/project/echo-embeddings/mteb_results/mistral/no_model_name_available/no_revision_available/", help="Path to the directory containing JSON logs")
args = parser.parse_args()

dir_path = args.dir_path


main_score = 0.0
count = 0

data = []

for json_log in os.listdir(dir_path):
    if json_log.endswith(".json") and json_log != "model_meta.json":
        with open(os.path.join(dir_path, json_log), "r") as f:
            log_data = json.load(f)
            main_score += log_data["scores"]["test"][0]["main_score"]
            data.append(
                {
                    "task_name": json_log,
                    "main_score": log_data["scores"]["test"][0]["main_score"],
                    "task": next((cat for cat, tasks in by_type.items() if log_data["task_name"] in tasks), None)
                }
            )
            count += 1

df = pd.DataFrame(data)

df_task = df.groupby("task")["main_score"].mean().reset_index()

print(df_task)

df_task.to_csv(os.path.join(dir_path, "summary.csv"), index=False)

