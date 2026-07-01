import os
import re
import time
import numpy as np
import pandas as pd

from models.meta_learners import fit_slearner
from evaluation.evaluate_models import evaluate

ROOT = "data"
RESULTS_DIR = "results/slearner"

os.makedirs(RESULTS_DIR, exist_ok=True)

results_all = []

for path, dirs, files in os.walk(ROOT):

    for file in files:

        if not file.endswith(".csv"):
            continue

        dataset_path = os.path.join(path, file)
        print(f"Procesando {dataset_path}")

        data = pd.read_csv(dataset_path)

        X = data[["x1", "x2", "x3"]].values
        W = data["treatment"].values
        Y = data["y"].values
        cate_true = data["cate_true"].values

        scenario = os.path.relpath(path, ROOT).replace("\\", "_")

        match = re.search(r"dataset_n(\d+)", file)
        n = int(match.group(1)) if match else -1

        rep = re.search(r"rep(\d+)", file)
        replication = int(rep.group(1)) if rep else 1

        start = time.perf_counter()

        model, cate_pred = fit_slearner(X, W, Y)

        end = time.perf_counter()

        execution_time = end - start

        cate_pred = np.asarray(cate_pred).reshape(-1)

        result = evaluate(
            cate_true,
            cate_pred,
            "SLearner",
            dataset_path
        )

        result["scenario"] = scenario
        result["n"] = n
        result["replication"] = replication
        result["execution_time"] = execution_time

        results_all.append(result)

if len(results_all) > 0:
    results_all = pd.concat(results_all, ignore_index=True)
    output_path = os.path.join(RESULTS_DIR, "slearner_results.csv")
    results_all.to_csv(output_path, index=False)
    print(f"\nResultados guardados en {output_path}")
else:
    print("\nNo se encontraron resultados para concatenar.")

print("\nSLearner finalizado.")