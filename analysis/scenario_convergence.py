import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================================
# Configuración
# ==========================================================

df = pd.read_csv(
    "results/all_results.csv"  # Cambiado para usar el mismo archivo que el código anterior
)

output = (
    "results/figures/convergence"
)

os.makedirs(
    output,
    exist_ok=True
)

# ==========================================================
# Definir los learners disponibles
# ==========================================================

learners = ["SNet", "TNet", "XNet", "DRNet"]

# ==========================================================
# Gráficas de convergencia para PEHE
# ==========================================================

for scenario in df["scenario"].unique():

    plt.figure(
        figsize=(8,6)
    )

    temp = df[
        df["scenario"] == scenario
    ]

    for learner in learners:  # Usar la lista definida en lugar de unique()

        sub = (

            temp[
                temp["learner"] == learner
            ]

            .groupby("n")["PEHE"]

            .agg(
                Mean="mean",
                SD="std",
                N="count"
            )

            .reset_index()

        )

        # Calcular error estándar e intervalos de confianza
        sub["SE"] = sub["SD"] / np.sqrt(sub["N"])
        sub["CI95_lower"] = sub["Mean"] - 1.96 * sub["SE"]
        sub["CI95_upper"] = sub["Mean"] + 1.96 * sub["SE"]

        # Línea principal con la media
        plt.plot(
            sub["n"],
            sub["Mean"],
            marker="o",
            label=learner,
            linewidth=2
        )

        # Banda de confianza del 95%
        plt.fill_between(
            sub["n"],
            sub["CI95_lower"],
            sub["CI95_upper"],
            alpha=0.2
        )

    plt.title(
        f"Convergencia PEHE - {scenario}"
    )

    plt.xlabel(
        "Sample Size (n)"
    )

    plt.ylabel(
        "PEHE"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        f"{output}/pehe_convergence_{scenario}.png",
        dpi=300
    )

    plt.close()

# ==========================================================
# Gráficas de convergencia para BIAS
# ==========================================================

for scenario in df["scenario"].unique():

    plt.figure(
        figsize=(8,6)
    )

    temp = df[
        df["scenario"] == scenario
    ]

    for learner in learners:

        sub = (

            temp[
                temp["learner"] == learner
            ]

            .groupby("n")["BIAS"]

            .agg(
                Mean="mean",
                SD="std",
                N="count"
            )

            .reset_index()

        )

        # Calcular error estándar e intervalos de confianza
        sub["SE"] = sub["SD"] / np.sqrt(sub["N"])
        sub["CI95_lower"] = sub["Mean"] - 1.96 * sub["SE"]
        sub["CI95_upper"] = sub["Mean"] + 1.96 * sub["SE"]

        # Línea principal con la media
        plt.plot(
            sub["n"],
            sub["Mean"],
            marker="o",
            label=learner,
            linewidth=2
        )

        # Banda de confianza del 95%
        plt.fill_between(
            sub["n"],
            sub["CI95_lower"],
            sub["CI95_upper"],
            alpha=0.2
        )

    plt.title(
        f"Convergencia BIAS - {scenario}"
    )

    plt.xlabel(
        "Sample Size (n)"
    )

    plt.ylabel(
        "BIAS"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        f"{output}/bias_convergence_{scenario}.png",
        dpi=300
    )

    plt.close()

print("Gráficas de convergencia generadas para PEHE y BIAS")