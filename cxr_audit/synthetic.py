import numpy as np
import pandas as pd


def make_predictions(n=6000, seed=42, labels=("Cardiomegaly", "Edema", "Pleural Effusion")):
    # synthetic model outputs with a built-in performance gap for one subgroup. Smoke test only.
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "sex": rng.choice(["F", "M"], n),
        "age_bin": rng.choice(["<40", "40-65", ">65"], n, p=[0.2, 0.4, 0.4]),
        "race": rng.choice(["White", "Black", "Asian", "Other"], n, p=[0.6, 0.2, 0.1, 0.1]),
    })
    for lab in labels:
        y = (rng.random(n) < 0.25).astype(int)
        signal = np.full(n, 2.0)
        signal[df.sex.values == "F"] -= 0.5
        signal[df.race.values == "Black"] -= 0.7
        signal[df.age_bin.values == ">65"] -= 0.3
        logit = (2 * y - 1) * signal / 2 + rng.normal(0, 1, n)
        df[f"y_{lab}"] = y
        df[f"p_{lab}"] = 1 / (1 + np.exp(-logit))
    return df
