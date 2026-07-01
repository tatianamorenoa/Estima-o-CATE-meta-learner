import os
import pandas as pd

# ======================================================
# Directorios
# ======================================================

RESULTS_DIR = "results"
OUTPUT_DIR = "analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================
# Archivos generados por los experimentos
# ======================================================

FILES = {
    "SNet": os.path.join(RESULTS_DIR, "slearner", "slearner_results.csv"),
    "TNet": os.path.join(RESULTS_DIR, "tlearner", "tlearner_results.csv"),
    "XNet": os.path.join(RESULTS_DIR, "xlearner", "xlearner_results.csv"),
    "DRNet": os.path.join(RESULTS_DIR, "drlearner", "drlearner_results.csv")
}

dfs = []

for learner, file in FILES.items():

    if not os.path.exists(file):
        print(f"[AVISO] No existe {file}")
        continue

    df = pd.read_csv(file)

    # Sobrescribe el nombre del learner para mantener consistencia
    df["learner"] = learner

    dfs.append(df)

if len(dfs) == 0:
    raise ValueError("No se encontraron archivos de resultados.")

# ======================================================
# Unir resultados
# ======================================================

results = pd.concat(dfs, ignore_index=True)

# ======================================================
# Ordenar
# ======================================================

results = results.sort_values(
    by=[
        "scenario",
        "n",
        "learner",
        "replication"
    ]
)

results.reset_index(drop=True, inplace=True)

# ======================================================
# Guardar
# ======================================================

OUTPUT_DIR = "results"

output_file = os.path.join(
    OUTPUT_DIR,
    "all_results.csv"
)

results.to_csv(
    output_file,
    index=False
)

print("=" * 60)
print("Resultados unidos correctamente")
print(f"Total observaciones : {len(results)}")
print(f"Archivo generado    : {output_file}")
print("=" * 60)

print(results.head())