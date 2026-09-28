"""Write data/frontier/full_set.json: the test items not yet in eval_set.json, so that the
full held-out sets can be evaluated.

  cbrest   ClaimBuster 2016 sentences outside the 800-item subsample  (800 + 1,945 = 2,745, all of 2016)
  clefdev  CLEF 2024 Task 1 EN dev split (1,032; unused so far, calib comes from train only)
  clefoff  CLEF 2024 Task 1 EN official test with gold labels (341), from
           https://gitlab.com/checkthat_lab/clef2024-checkthat-lab (task1/data/CT24_checkworthy_test_gold.zip)

Ids are positions in each list; score them with scripts/build_frontier_prompts.py full_set.json.
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent


def rows(df):
    return [dict(Sentence_id=str(r.Sentence_id), text=r.text, label=int(r.label)) for r in df.itertuples()]


def main():
    ev = json.loads((HERE / "frontier" / "eval_set.json").read_text(encoding="utf-8"))
    cb = pd.read_csv(HERE / "processed" / "claimbuster_test.csv")
    have = {r["Sentence_id"] for r in ev["cb"]}
    cbrest = cb[~cb.Sentence_id.astype(str).isin(have)]
    gold = pd.read_csv(HERE / "raw" / "clef2024" / "official_test_gold.tsv", sep="\t")
    gold = gold.assign(text=gold.Text, label=(gold.class_label == "Yes").astype(int))
    gold.drop(columns=["Text", "class_label"]).to_csv(HERE / "processed" / "clef_official_test.csv", index=False)
    out = dict(cbrest=rows(cbrest), clefdev=rows(pd.read_csv(HERE / "processed" / "clef_dev.csv")),
               clefoff=rows(gold))
    (HERE / "frontier" / "full_set.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print({k: (len(v), sum(r["label"] for r in v)) for k, v in out.items()})


if __name__ == "__main__":
    main()
