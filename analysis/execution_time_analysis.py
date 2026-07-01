import os
import numpy as np
import pandas as pd

# ==========================================================
# Configuración
# ==========================================================

INPUT_FILE = "results/all_results.csv"

OUTPUT_FILE = "results/execution_time_summary.csv"

# ==========================================================
# Leer resultados
# ==========================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"No existe el archivo:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

# ==========================================================
# Verificación
# ==========================================================

required_columns = [
    "scenario",
    "n",
    "learner",
    "execution_time"
]

missing = [c for c in required_columns if c not in df.columns]

if len(missing) > 0:
    raise ValueError(
        f"Faltan columnas:\n{missing}"
    )

# ==========================================================
# Estadísticas
# ==========================================================

summary = []

groups = df.groupby(
    [
        "scenario",
        "n",
        "learner"
    ]
)

for (scenario, n, learner), group in groups:

    x = group["execution_time"].dropna().values

    N = len(x)

    mean = np.mean(x)

    sd = np.std(x, ddof=1) if N > 1 else 0

    se = sd / np.sqrt(N) if N > 1 else 0

    ci_low = mean - 1.96 * se

    ci_high = mean + 1.96 * se

    summary.append({

        "scenario": scenario,

        "n": n,

        "learner": learner,

        "replications": N,

        "mean_time": mean,

        "std_time": sd,

        "se_time": se,

        "ci95_lower": ci_low,

        "ci95_upper": ci_high,

        "min_time": np.min(x),

        "median_time": np.median(x),

        "max_time": np.max(x)

    })

summary = pd.DataFrame(summary)

summary.sort_values(

    by=[

        "scenario",

        "n",

        "learner"

    ],

    inplace=True

)

summary.reset_index(drop=True, inplace=True)

# ==========================================================
# Guardar
# ==========================================================

summary.to_csv(

    OUTPUT_FILE,

    index=False

)

# ==========================================================
# Mostrar resultados
# ==========================================================

print("=" * 70)

print("EXECUTION TIME SUMMARY")

print("=" * 70)

print(summary)

print()

print(f"Archivo generado: {OUTPUT_FILE}")

print("=" * 70)

# ==========================================================
# Tabla comparativa
# ==========================================================

comparison = summary.pivot_table(

    index=[

        "scenario",

        "n"

    ],

    columns="learner",

    values="mean_time"

)

print()

print("=" * 70)

print("COMPARACIÓN DEL TIEMPO MEDIO (segundos)")

print("=" * 70)

print(comparison.round(4))