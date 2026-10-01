# Produces predictions.csv (columns: metadata + y_<label> + p_<label>) using a pretrained TorchXRayVision model.
# Written against the torchxrayvision API; verify on a few images before a full run.
# Usage: python -m cxr_audit.infer_txv --meta meta.csv --image-root /path/to/images --out predictions.csv
# `meta.csv` needs a column `image_path` (relative to --image-root) plus the metadata/label columns from the loaders.
import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="densenet121-res224-all")
    a = ap.parse_args()
    import skimage.io
    import torch
    import torchxrayvision as xrv

    model = xrv.models.DenseNet(weights=a.weights).eval()
    meta = pd.read_csv(a.meta)
    labels = [c[2:] for c in meta.columns if c.startswith("y_")]
    probs = {l: [] for l in labels}
    for path in meta.image_path:
        img = skimage.io.imread(f"{a.image_root}/{path}")
        if img.ndim == 3:
            img = img.mean(2)
        img = xrv.datasets.normalize(img, 255)[None, ...]
        img = xrv.datasets.XRayCenterCrop()(img)
        img = xrv.datasets.XRayResizer(224)(img)
        with torch.no_grad():
            out = model(torch.from_numpy(img)[None, ...]).numpy()[0]
        for l in labels:
            probs[l].append(float(out[model.pathologies.index(l)]) if l in model.pathologies else np.nan)
    for l in labels:
        meta[f"p_{l}"] = probs[l]
    meta.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
