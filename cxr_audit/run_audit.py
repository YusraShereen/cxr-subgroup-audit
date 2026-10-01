import argparse
import os

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .audit import run_audit  # noqa: E402
from .synthetic import make_predictions  # noqa: E402
from .utils import md_table, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    set_seed(cfg["seed"])
    out = cfg["out_dir"]
    os.makedirs(out, exist_ok=True)
    df = make_predictions(seed=cfg["seed"], labels=cfg["labels"]) if cfg.get("synthetic") else pd.read_csv(cfg["predictions_csv"])

    groups, gaps = run_audit(df, cfg["labels"], cfg["attributes"], cfg["n_boot"], cfg["min_group_n"], cfg["seed"])
    groups.to_csv(os.path.join(out, "subgroup_metrics.csv"), index=False)
    gaps.to_csv(os.path.join(out, "gap_summary.csv"), index=False)
    gaps_md = gaps.copy()
    gaps_md["ci_excludes_0"] = gaps_md.gap_ci_lo > 0.0
    with open(os.path.join(out, "audit_report.md"), "w") as f:
        f.write("# Subgroup audit report\n\n## AUROC / calibration gaps (max - min across groups)\n\n")
        f.write(md_table(gaps_md))
        f.write("\nNote: the gap CI is a bootstrap over groups; the max-min statistic is biased upward in small groups, "
                "so compare gaps against the CI and group sizes rather than reading them in isolation.\n\n")
        f.write("## Per-group metrics\n\n" + md_table(groups))

    attrs = cfg["attributes"]
    fig, axes = plt.subplots(1, len(attrs), figsize=(5 * len(attrs), 4), squeeze=False)
    for ax, at in zip(axes[0], attrs):
        sub = groups[groups.attribute == at]
        labs = list(dict.fromkeys(sub.label))
        gs = list(dict.fromkeys(sub.group))
        w = 0.8 / len(gs)
        for i, g in enumerate(gs):
            vals = [sub[(sub.label == l) & (sub.group == g)].auroc.mean() for l in labs]
            ax.bar(np.arange(len(labs)) + i * w, vals, w, label=str(g))
        ax.set_xticks(np.arange(len(labs)) + 0.4 - w / 2, labs, rotation=15)
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel("AUROC")
        ax.set_title(f"By {at}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "auroc_by_group.png"), dpi=200)
    print(md_table(gaps_md[["label", "attribute", "auroc_gap", "gap_ci_lo", "gap_ci_hi", "worst_group", "ci_excludes_0"]]))


if __name__ == "__main__":
    main()
