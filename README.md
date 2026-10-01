# cxr-subgroup-audit

Subgroup performance and calibration audit of a chest X-ray classifier.

**Research question:** Does the model's performance differ across patient subgroups?

**Paper:** _arXiv link — TBD (add here as soon as the preprint is posted)_

## Goal

Audit a chest X-ray model on a public dataset with demographic labels. Report subgroup AUROC gaps (with bootstrap CIs) and calibration by group, plus TPR/FPR gaps at a common operating point.

## Data

- **MIMIC-CXR-JPG + MIMIC-IV** (sex, age, race): requires credentialed PhysioNet access and the CITI training course. **Start this early.**
- **CheXpert** (sex, age; public CheXpert has no race field): use as the fallback if MIMIC access is slow. Do not redistribute either dataset.
- `data_mimic.py` / `data_chexpert.py` build the label + demographic tables.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run (one command)

```bash
make run        # audit data/predictions.csv -> results/
make smoke      # synthetic sanity check
```

`make smoke` runs the whole pipeline end-to-end on synthetic data in under a minute and checks that the code works. The numbers it produces are NOT results.

All experiments use fixed seeds (`seed: 42` in the configs, applied through `utils.set_seed`). After your final run, lock the environment with `pip freeze > requirements.lock.txt`.

## Repository layout

```
cxr_audit/
  data_chexpert.py  CheXpert metadata + labels
  data_mimic.py     MIMIC-CXR + MIMIC-IV join (sex, age, race)
  infer_txv.py      predictions with a pretrained TorchXRayVision model
  audit.py          subgroup AUROC/ECE/TPR/FPR, bootstrap gaps
  run_audit.py      report + figure
configs/ results/ paper/
```

## Results

### Gap summary (TBD after real runs)

| label | attribute | AUROC gap | 95% CI | worst group | ECE gap | TPR gap | FPR gap |
|---|---|---|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Main figure

![auroc by group](results/auroc_by_group.png)

## Limitations

- Race labels in MIMIC are administrative and noisy; frame findings as disparities in recorded categories.
- Small groups inflate max-min gaps; always read them with the CI and group sizes.
- Uncertain labels are mapped with U-Zeros by default (`uncertain` option).

## Citation

See `CITATION.cff`. A BibTeX entry will be added once the arXiv paper is out.
