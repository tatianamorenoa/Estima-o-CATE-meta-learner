import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# Configuración
# ==========================================================

INPUT_FILE = "results/all_results.csv"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

# ==========================================================
# Carpetas de salida
# ==========================================================

folders = {
    "SNet": "results/slearner",
    "TNet": "results/tlearner",
    "XNet": "results/xlearner",
    "DRNet": "results/drlearner"
}

# ==========================================================
# Análisis por learner
# ==========================================================

for learner, folder in folders.items():

    print(f"\nProcesando {learner}")

    data = df[df["learner"] == learner]

    # ======================================================
    # Resumen PEHE
    # ======================================================

    summary_pehe = (

        data

        .groupby(

            ["scenario", "n"]

        )["PEHE"]

        .agg(

            Mean="mean",
            SD="std",
            N="count"

        )

        .reset_index()

    )

    summary_pehe["SE"] = summary_pehe["SD"] / np.sqrt(summary_pehe["N"])

    summary_pehe["CI95_lower"] = summary_pehe["Mean"] - 1.96 * summary_pehe["SE"]

    summary_pehe["CI95_upper"] = summary_pehe["Mean"] + 1.96 * summary_pehe["SE"]

    # ======================================================
    # Resumen BIAS
    # ======================================================

    summary_bias = (

        data

        .groupby(

            ["scenario", "n"]

        )["BIAS"]

        .agg(

            Mean="mean",
            SD="std",
            N="count"

        )

        .reset_index()

    )

    summary_bias["SE"] = summary_bias["SD"] / np.sqrt(summary_bias["N"])

    summary_bias["CI95_lower"] = summary_bias["Mean"] - 1.96 * summary_bias["SE"]

    summary_bias["CI95_upper"] = summary_bias["Mean"] + 1.96 * summary_bias["SE"]

    # ======================================================
    # Guardar tablas
    # ======================================================

    output_csv_pehe = os.path.join(
        folder,
        "pehe_summary.csv"
    )

    summary_pehe.to_csv(
        output_csv_pehe,
        index=False
    )

    output_csv_bias = os.path.join(
        folder,
        "bias_summary.csv"
    )

    summary_bias.to_csv(
        output_csv_bias,
        index=False
    )

    # ======================================================
    # Heatmap PEHE
    # ======================================================

    heat_pehe = summary_pehe.pivot(
        index="scenario",
        columns="n",
        values="Mean"
    )

    fig, ax = plt.subplots(figsize=(8,5))

    im = ax.imshow(
        heat_pehe.values,
        aspect="auto"
    )

    ax.set_xticks(np.arange(len(heat_pehe.columns)))
    ax.set_xticklabels(heat_pehe.columns)

    ax.set_yticks(np.arange(len(heat_pehe.index)))
    ax.set_yticklabels(heat_pehe.index)

    ax.set_xlabel("Sample size")
    ax.set_ylabel("Scenario")
    ax.set_title(f"{learner} - Mean PEHE")

    for i in range(heat_pehe.shape[0]):
        for j in range(heat_pehe.shape[1]):
            ax.text(
                j,
                i,
                f"{heat_pehe.iloc[i,j]:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white"
            )

    plt.colorbar(
        im,
        label="Mean PEHE"
    )

    plt.tight_layout()

    output_png_pehe = os.path.join(
        folder,
        "pehe_heatmap.png"
    )

    plt.savefig(
        output_png_pehe,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ======================================================
    # Heatmap BIAS
    # ======================================================

    heat_bias = summary_bias.pivot(
        index="scenario",
        columns="n",
        values="Mean"
    )

    fig, ax = plt.subplots(figsize=(8,5))

    im = ax.imshow(
        heat_bias.values,
        aspect="auto"
    )

    ax.set_xticks(np.arange(len(heat_bias.columns)))
    ax.set_xticklabels(heat_bias.columns)

    ax.set_yticks(np.arange(len(heat_bias.index)))
    ax.set_yticklabels(heat_bias.index)

    ax.set_xlabel("Sample size")
    ax.set_ylabel("Scenario")
    ax.set_title(f"{learner} - Mean BIAS")

    for i in range(heat_bias.shape[0]):
        for j in range(heat_bias.shape[1]):
            ax.text(
                j,
                i,
                f"{heat_bias.iloc[i,j]:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white"
            )

    plt.colorbar(
        im,
        label="Mean BIAS"
    )

    plt.tight_layout()

    output_png_bias = os.path.join(
        folder,
        "bias_heatmap.png"
    )

    plt.savefig(
        output_png_bias,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Guardado: {output_csv_pehe}")
    print(f"Guardado: {output_png_pehe}")
    print(f"Guardado: {output_csv_bias}")
    print(f"Guardado: {output_png_bias}")

print("\nProceso finalizado.")