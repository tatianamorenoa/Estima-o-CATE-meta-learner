import os

from simulation.data_simulation import (
    simular_dataset
)

R = 20

base_path = "data/balanced"

# =========================
# WITH CORRELATION
# =========================

corr_path = os.path.join(
    base_path,
    "with_correlation"
)

os.makedirs(
    corr_path,
    exist_ok=True
)

for rep in range(R):

    for n in [100, 2000, 5000]:

        data = simular_dataset(
            n=n,
            v=(0.2, 0.5, 0.3),
            prob=0.5,
            seed=1000 + rep
        )

        data.to_csv(

            os.path.join(
                corr_path,
                f"dataset_n{n}_rep{rep+1}.csv"
            ),

            index=False
        )

# =========================
# WITHOUT CORRELATION
# =========================

no_corr_path = os.path.join(
    base_path,
    "without_correlation"
)

os.makedirs(
    no_corr_path,
    exist_ok=True
)

for rep in range(R):

    for n in [100, 2000, 5000]:

        data = simular_dataset(
            n=n,
            v=(0.0, 0.0, 0.0),
            prob=0.5,
            seed=5000 + rep
        )

        data.to_csv(

            os.path.join(
                no_corr_path,
                f"dataset_n{n}_rep{rep+1}.csv"
            ),

            index=False
        )

print("Balanced Monte Carlo finalizado")