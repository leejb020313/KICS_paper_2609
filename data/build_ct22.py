"""Add CheckThat! 2022 Task 1A English (check-worthiness of COVID-19 tweets) as a third dataset.

Source: https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab (task1/data/subtasks-english,
CT22_english_1A_checkworthy.zip and test/CT22_english_1A_checkworthy_test_gold.zip), unzipped into data/raw/ct22/.

Same split logic as CLEF 2024 in build_datasets.py: the learning ("calibration") set is a class-balanced
subsample of train (all 447 positives + 447 negatives), and the test set is every held-out split
(dev + dev_test + test gold). Writes the keys ct22 / ct22cal into data/frontier/{eval,calib}_set.json,
which is where the frontier-LLM prompts and final_eval.py read them from.
"""
import json
from pathlib import Path

import pandas as pd

RAW = Path(__file__).parent / "raw" / "ct22"
FRONTIER = Path(__file__).parent / "frontier"
SEED = 42


def load(split):
    df = pd.read_csv(RAW / f"CT22_english_1A_checkworthy_{split}.tsv", sep="\t", dtype={"tweet_id": str})
    # one line per statement in the batched prompt: flatten newlines and tabs inside tweets
    text = df["tweet_text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return pd.DataFrame({"Sentence_id": df["tweet_id"], "text": text, "label": df["class_label"].astype(int)})


def main():
    train = load("train")
    pos = train[train["label"] == 1]
    neg = train[train["label"] == 0].sample(n=len(pos), random_state=SEED)
    calib = pd.concat([pos, neg]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    test = pd.concat([load(s) for s in ("dev_test", "dev", "test_gold")], ignore_index=True)

    for fn, key, df in (("calib_set.json", "ct22cal", calib), ("eval_set.json", "ct22", test)):
        path = FRONTIER / fn
        sets = json.loads(path.read_text(encoding="utf-8"))
        sets[key] = df.to_dict("records")
        path.write_text(json.dumps(sets, ensure_ascii=False), encoding="utf-8")  # same serialisation as the existing files
    print(f"CT22 EN 1A: calib={len(calib)} (pos={calib['label'].sum()}), test={len(test)} (pos={test['label'].sum()})")


if __name__ == "__main__":
    main()
