import os
import numpy as np
import pandas as pd

# ==========================================================
# Configuración
# ==========================================================

INPUT_FILE = "results/all_results.csv"
OUTPUT_FILE = "results/summary_statistics.csv"

# ==========================================================
# Leer resultados
# ==========================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"No existe el archivo:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

# ==========================================================
# Variables de interés
# ==========================================================

metrics = [
    "PEHE",
    "BIAS",
    "RMSE",
    "execution_time"
]

group_columns = [
    "scenario",
    "n",
    "learner"
]

summary = []

# ==========================================================
# Calcular estadísticas
# ==========================================================

for metric in metrics:

    grouped = df.groupby(group_columns)

    for name, group in grouped:

        values = group[metric].dropna().values

        N = len(values)

        mean = np.mean(values)

        sd = np.std(values, ddof=1) if N > 1 else 0

        se = sd / np.sqrt(N) if N > 1 else 0

        ci_lower = mean - 1.96 * se

        ci_upper = mean + 1.96 * se

        summary.append({

            "scenario": name[0],

            "n": name[1],

            "learner": name[2],

            "metric": metric,

            "N": N,

            "Mean": mean,

            "SD": sd,

            "SE": se,

            "CI95_lower": ci_lower,

            "CI95_upper": ci_upper

        })

# ==========================================================
# DataFrame final
# ==========================================================

summary = pd.DataFrame(summary)

summary = summary.sort_values(

    by=[

        "scenario",

        "n",

        "metric",

        "learner"

    ]

)

summary.reset_index(

    drop=True,

    inplace=True

)

# ==========================================================
# Guardar
# ==========================================================

summary.to_csv(

    OUTPUT_FILE,

    index=False

)

# ==========================================================
# Mostrar resumen
# ==========================================================

print("=" * 70)

print("SUMMARY STATISTICS")

print("=" * 70)

print(summary.head(20))

print()

print(f"Total filas : {len(summary)}")

print(f"Archivo generado : {OUTPUT_FILE}")

print("=" * 70)