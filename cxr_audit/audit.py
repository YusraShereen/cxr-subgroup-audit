import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .utils import ece_binary


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def _auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


def _rates(y, p, thr):
    pred = p >= thr
    tpr = pred[y == 1].mean() if (y == 1).any() else np.nan
    fpr = pred[y == 0].mean() if (y == 0).any() else np.nan
    return tpr, fpr


def audit_label_attr(df, label, attr, n_boot=200, min_n=50, seed=42):
    # returns (per-group table, gap summary row) for one label and one demographic attribute
    rng = np.random.default_rng(seed)
    d = df[df[f"y_{label}"].isin([0, 1])].dropna(subset=[attr])
    y_all, p_all = d[f"y_{label}"].values.astype(int), d[f"p_{label}"].values
    thr = youden_threshold(y_all, p_all)
    rows, boots = [], {}
    for g, dg in d.groupby(attr):
        y, p = dg[f"y_{label}"].values.astype(int), dg[f"p_{label}"].values
        if len(dg) < min_n or len(np.unique(y)) < 2:
            continue
        tpr, fpr = _rates(y, p, thr)
        b = []
        for _ in range(n_boot):
            i = rng.integers(0, len(dg), len(dg))
            b.append(_auc(y[i], p[i]))
        boots[g] = np.array(b)
        lo, hi = np.nanpercentile(b, [2.5, 97.5])
        rows.append(dict(label=label, attribute=attr, group=g, n=len(dg), prevalence=y.mean(),
                         auroc=_auc(y, p), auroc_lo=lo, auroc_hi=hi, ece=ece_binary(p, y), tpr=tpr, fpr=fpr))
    tab = pd.DataFrame(rows)
    if len(tab) < 2:
        return tab, None
    gaps = np.nanmax(list(boots.values()), 0) - np.nanmin(list(boots.values()), 0)
    gap = dict(label=label, attribute=attr, auroc_gap=tab.auroc.max() - tab.auroc.min(),
               gap_ci_lo=float(np.nanpercentile(gaps, 2.5)), gap_ci_hi=float(np.nanpercentile(gaps, 97.5)),
               worst_group=tab.loc[tab.auroc.idxmin(), "group"], best_group=tab.loc[tab.auroc.idxmax(), "group"],
               ece_gap=tab.ece.max() - tab.ece.min(), tpr_gap=tab.tpr.max() - tab.tpr.min(), fpr_gap=tab.fpr.max() - tab.fpr.min())
    return tab, gap


def run_audit(df, labels, attrs, n_boot=200, min_n=50, seed=42):
    tabs, gaps = [], []
    for lab in labels:
        for at in attrs:
            t, g = audit_label_attr(df, lab, at, n_boot, min_n, seed)
            tabs.append(t)
            if g:
                gaps.append(g)
    return pd.concat(tabs, ignore_index=True), pd.DataFrame(gaps)
