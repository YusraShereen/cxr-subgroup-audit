import os
import random

import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def reliability_bins(conf, correct, n_bins=10):
    # returns list of (mean_conf, accuracy, count) per equal-width bin
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
        else:
            out.append((float("nan"), float("nan"), 0))
    return out


def ece(conf, correct, n_bins=10):
    # expected calibration error: confidence of the predicted answer vs. its accuracy
    n = len(conf)
    if n == 0:
        return float("nan")
    return float(sum(c / n * abs(a - cf) for cf, a, c in reliability_bins(conf, correct, n_bins) if c > 0))


def ece_binary(p, y, n_bins=10):
    # binary ECE: predicted P(y=1) vs. observed frequency of y=1
    return ece(p, y, n_bins)


def plot_reliability(curves, path, title="Reliability diagram", n_bins=10):
    # curves: dict name -> (confidences, correct)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, (conf, correct) in curves.items():
        if len(conf) == 0:
            continue
        bins = reliability_bins(conf, correct, n_bins)
        xs = [b[0] for b in bins if b[2] > 0]
        ys = [b[1] for b in bins if b[2] > 0]
        ax.plot(xs, ys, marker="o", label=f"{name} (ECE={ece(conf, correct, n_bins):.3f})")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(fontsize=7)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
