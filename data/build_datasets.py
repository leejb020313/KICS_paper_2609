"""Rebuild clean, paper-matching splits from data/raw into data/processed.

ClaimBuster source: https://zenodo.org/records/3836810 (ClaimBuster_Datasets.zip)
CLEF 2024 CheckThat! Task 1 (EN) source: https://huggingface.co/datasets/iai-group/clef2024_checkthat_task1_en

Reproduces the NN-PPI paper's (arXiv 2608.30731) split logic:
- ClaimBuster: calibration = 2012 debates (class-balanced subsample), test = 2016 debates (full)
- CLEF 2024: calibration = class-balanced subsample of train, dev/test = official splits
"""
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(__file__).parent / "raw"
OUT = Path(__file__).parent / "processed"
OUT.mkdir(exist_ok=True)

SEED = 42


def build_claimbuster():
    z = zipfile.ZipFile(RAW / "ClaimBuster_Datasets.zip")
    gt = pd.read_csv(io.BytesIO(z.read("ClaimBuster_Datasets/datasets/groundtruth.csv")))
    cs = pd.read_csv(io.BytesIO(z.read("ClaimBuster_Datasets/datasets/crowdsourced.csv")))
    full = pd.concat([gt, cs], ignore_index=True)
    full["year"] = full["File_id"].astype(str).str.slice(0, 4)
    # Verdict: -1=NFS, 0=UFS, 1=CFS. Binary check-worthy = Verdict==1.
    full["label"] = (full["Verdict"] == 1).astype(int)

    calib_pool = full[full["year"] == "2012"].copy()
    test = full[full["year"] == "2016"].copy()

    # class-balanced calibration subsample, paper reports |L|=1314
    n_per_class = 1314 // 2
    pos = calib_pool[calib_pool["label"] == 1].sample(
        n=min(n_per_class, (calib_pool["label"] == 1).sum()), random_state=SEED
    )
    neg = calib_pool[calib_pool["label"] == 0].sample(
        n=min(n_per_class, (calib_pool["label"] == 0).sum()), random_state=SEED
    )
    calib = pd.concat([pos, neg]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    calib[["Sentence_id", "Text", "label"]].rename(columns={"Text": "text"}).to_csv(
        OUT / "claimbuster_calib.csv", index=False
    )
    test[["Sentence_id", "Text", "label"]].rename(columns={"Text": "text"}).to_csv(
        OUT / "claimbuster_test.csv", index=False
    )
    print(f"ClaimBuster: calib={len(calib)} (pos={calib['label'].sum()}), "
          f"test={len(test)} (pos={test['label'].sum()}, neg={(test['label']==0).sum()})")


def build_clef():
    def load(split):
        df = pd.read_csv(RAW / "clef2024" / f"{split}.tsv", sep="\t")
        df["label"] = (df["class_label"] == "Yes").astype(int)
        return df[["Sentence_id", "Text", "label"]].rename(columns={"Text": "text"})

    train = load("train")
    dev = load("dev")
    test = load("test")

    n_per_class = 2406 // 2
    pos = train[train["label"] == 1].sample(n=n_per_class, random_state=SEED)
    neg = train[train["label"] == 0].sample(n=n_per_class, random_state=SEED)
    calib = pd.concat([pos, neg]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    calib.to_csv(OUT / "clef_calib.csv", index=False)
    dev.to_csv(OUT / "clef_dev.csv", index=False)
    test.to_csv(OUT / "clef_test.csv", index=False)
    print(f"CLEF 2024: calib={len(calib)} (pos={calib['label'].sum()}), "
          f"test={len(test)} (pos={test['label'].sum()}, neg={(test['label']==0).sum()})")


if __name__ == "__main__":
    build_claimbuster()
    build_clef()
