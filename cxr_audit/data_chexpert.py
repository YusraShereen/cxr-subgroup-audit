import pandas as pd

LABELS = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]


def load_chexpert(csv_path, uncertain="zeros"):
    # CheXpert (small) CSV -> frame with sex, age_bin and binary labels. Public CheXpert has no race field.
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Sex": "sex", "Age": "age"})
    df["sex"] = df["sex"].map({"Female": "F", "Male": "M"})
    df["age_bin"] = pd.cut(df["age"], [0, 40, 65, 120], labels=["<40", "40-65", ">65"], right=False).astype(str)
    for lab in LABELS:
        s = df[lab]
        s = s.where(s != -1, 0 if uncertain == "zeros" else 1)   # U-Zeros / U-Ones
        df[f"y_{lab}"] = s.fillna(0).astype(int)
    return df[["Path", "sex", "age_bin"] + [f"y_{l}" for l in LABELS]]
