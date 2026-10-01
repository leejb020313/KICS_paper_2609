"""Checks behind three side statements in the paper.

1. Batched vs one-by-one frontier scoring (Section III): 80 CLEF test claims were also scored one
   at a time with the few-shot prompt (results/llm_scores/checks/single). 11 calls hit the CLI usage limit
   and returned no score, leaving 69; ROC-AUC on those 69 is compared with the batched scores.
2. Stability of batched scoring: a second batched pass over CLEF test (results/llm_scores/checks/rerun),
   Pearson correlation with the first pass.
3. SVM inference latency on CPU (Section II, "about 6 ms per sentence"), embedding included.
   Machine-dependent; not part of verify_paper_numbers.py.
"""
import os
import re
import sys
import time

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cwcascade.data import DATA, SCORES, embed, load_frontier, read_frontier_scores, read_json  # noqa: E402


def single_scores():
    folder = os.path.join(SCORES, 'checks', 'single')
    out = {}
    for i in read_json(os.path.join(folder, 'ids.json')):
        with open(os.path.join(folder, f'q_{i:04d}.out'), encoding='utf-8') as f:
            m = re.search(r'"confidence_score"\s*:\s*([0-9.]+)', f.read())
        if m:
            out[i] = float(m.group(1))
    return out


def main():
    items = read_json(os.path.join(DATA, 'frontier', 'eval_set.json'))['clef']
    y = np.array([r['label'] for r in items])
    batched = read_frontier_scores()['clef']

    single = single_scores()
    ids = sorted(single)
    print(f"[1] batched vs single, CLEF n={len(ids)}: AUC single={roc_auc_score(y[ids], [single[i] for i in ids]):.3f}"
          f"  batched={roc_auc_score(y[ids], [batched[i] for i in ids]):.3f}")

    rerun = read_frontier_scores(os.path.join('checks', 'rerun'))['clef']
    ids = sorted(rerun)
    r = pearsonr([batched[i] for i in ids], [rerun[i] for i in ids])[0]
    print(f"[2] batched pass 1 vs pass 2, CLEF n={len(ids)}: Pearson r={r:.3f}")

    test, calib, test_items = load_frontier('cb')
    svm = SVC(class_weight='balanced', random_state=0).fit(calib.X, calib.y)
    texts = [r['text'] for r in test_items]
    embed(texts[:8])  # warm-up
    t0 = time.perf_counter()
    svm.decision_function(embed(texts))
    print(f"[3] embedding + SVM on CPU: {1000 * (time.perf_counter() - t0) / len(texts):.1f} ms/sentence (n={len(texts)})")


if __name__ == '__main__':
    main()
