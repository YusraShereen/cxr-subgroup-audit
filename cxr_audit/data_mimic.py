import pandas as pd

LABELS = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]


def load_mimic(mimic_cxr_dir, mimic_iv_dir, uncertain="zeros"):
    # Joins MIMIC-CXR-JPG labels/metadata with MIMIC-IV demographics (credentialed PhysioNet access required).
    #   <mimic_cxr_dir>/mimic-cxr-2.0.0-metadata.csv(.gz) and mimic-cxr-2.0.0-chexpert.csv(.gz)
    #   <mimic_iv_dir>/hosp/patients.csv(.gz) and hosp/admissions.csv(.gz)
    # Adjust file names to the versions you downloaded.
    import os
    def rd(d, name):
        for ext in ("", ".gz"):
            p = os.path.join(d, name + ext)
            if os.path.exists(p):
                return pd.read_csv(p)
        raise FileNotFoundError(name)

    meta = rd(mimic_cxr_dir, "mimic-cxr-2.0.0-metadata.csv")
    labs = rd(mimic_cxr_dir, "mimic-cxr-2.0.0-chexpert.csv")
    pat = rd(mimic_iv_dir, "hosp/patients.csv")
    adm = rd(mimic_iv_dir, "hosp/admissions.csv")
    race = adm.sort_values("admittime").groupby("subject_id").race.last().reset_index()
    race["race"] = race.race.str.upper().map(
        lambda r: "White" if "WHITE" in r else "Black" if "BLACK" in r else "Asian" if "ASIAN" in r else "Other")
    df = labs.merge(meta[["subject_id", "study_id", "dicom_id", "ViewPosition"]].drop_duplicates("study_id"), on=["subject_id", "study_id"])
    df = df.merge(pat[["subject_id", "gender", "anchor_age"]], on="subject_id").merge(race, on="subject_id", how="left")
    df["sex"] = df.gender
    df["age_bin"] = pd.cut(df.anchor_age, [0, 40, 65, 130], labels=["<40", "40-65", ">65"], right=False).astype(str)
    for lab in LABELS:
        s = df[lab]
        s = s.where(s != -1, 0 if uncertain == "zeros" else 1)
        df[f"y_{lab}"] = s.fillna(0).astype(int)
    return df[["subject_id", "study_id", "dicom_id", "sex", "age_bin", "race"] + [f"y_{l}" for l in LABELS]]
