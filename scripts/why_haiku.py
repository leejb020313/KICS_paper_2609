"""Why does Claude Haiku 4.5, called on every sentence, score higher accuracy than Claude Sonnet 5?

Same sentences, same prompt, same batches; only the model differs. Separates three explanations:
  (1) ranking  - does one model order check-worthy sentences above the rest better? (AUC)
  (2) scale    - is one model's 0.5 simply in the wrong place? (flag rate, label rate per score bin,
                 accuracy at 0.5 vs the best single threshold chosen on the test set itself)
  (3) transfer - does the threshold chosen on the class-balanced learning set carry over to the
                 imbalanced test set? (learning-set threshold vs test-oracle threshold)
and lists sentences on which the two models disagree, to see what kind of sentence it is.

Writes results/analysis/why_haiku.json and prints a report.
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cwcascade.data import DATA, DATASETS, FULL_PARTS, best_threshold, read_frontier_scores, read_json  # noqa: E402

LLMS = {"sonnet5": "Sonnet 5", "haiku45": "Haiku 4.5"}
BINS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0001]


def sets(name):
    """Test items (whole held-out set, as in the paper) and learning-set items, with ids into each score file."""
    test = [(name, i, r) for i, r in enumerate(read_json(os.path.join(DATA, "frontier", "eval_set.json"))[name])]
    extra = read_json(os.path.join(DATA, "frontier", "full_set.json"))
    for part in FULL_PARTS[name]:
        test += [(part, i, r) for i, r in enumerate(extra[part])]
    ck = DATASETS[name][0]
    calib = [(ck, i, r) for i, r in enumerate(read_json(os.path.join(DATA, "frontier", "calib_set.json"))[ck])]
    return test, calib


def scores(items, fs):
    keep = [(k, i, r) for k, i, r in items if i in fs.get(k, {})]
    return np.array([fs[k][i] for k, i, _ in keep]), np.array([r["label"] for *_, r in keep]), [r["text"] for *_, r in keep]


def acc(s, y, t):
    return float(np.mean((s >= t) == y))


out = {}
FS = {m: read_frontier_scores(m) for m in LLMS}
for name in ("clef", "cb"):
    test, calib = sets(name)
    out[name] = {}
    T = {}
    for m in LLMS:
        s, y, txt = scores(test, FS[m])
        sc, yc, _ = scores(calib, FS[m])
        t_learn = float(best_threshold(sc, yc))
        t_oracle = float(best_threshold(s, y))
        bins = []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            k = (s >= lo) & (s < hi)
            bins.append(dict(bin=f"[{lo:.1f},{min(hi, 1):.1f})", n=int(k.sum()), share=float(k.mean()),
                             label_rate=float(y[k].mean()) if k.any() else None))
        out[name][m] = dict(
            n_test=int(len(y)), test_pos_rate=float(y.mean()), learn_pos_rate=float(yc.mean()),
            auc_test=float(roc_auc_score(y, s)), auc_learn=float(roc_auc_score(yc, sc)),
            flag_rate_05=float(np.mean(s >= 0.5)),
            prec_05=float(y[s >= 0.5].mean()), rec_05=float(np.mean(s[y == 1] >= 0.5)),
            acc_05=acc(s, y, 0.5), t_learn=t_learn, acc_t_learn=acc(s, y, t_learn),
            t_oracle=t_oracle, acc_t_oracle=acc(s, y, t_oracle),
            mean_score_pos=float(s[y == 1].mean()), mean_score_neg=float(s[y == 0].mean()),
            n_distinct_scores=int(len(np.unique(s))), bins=bins)
        T[m] = (s, y, txt)
    # sentence-level agreement at each model's own 0.5 cut
    (ss, y, txt), (sh, _, _) = T["sonnet5"], T["haiku45"]
    assert len(ss) == len(sh)
    S, H = ss >= 0.5, sh >= 0.5
    dis = S != H
    out[name]["agreement"] = dict(
        both_yes=int((S & H).sum()), both_no=int((~S & ~H).sum()),
        haiku_only_yes=int((H & ~S).sum()), sonnet_only_yes=int((S & ~H).sum()),
        haiku_only_yes_label_rate=float(y[H & ~S].mean()), sonnet_only_yes_label_rate=float(y[S & ~H].mean()),
        disagree_haiku_right=int((dis & (H == y)).sum()), disagree_sonnet_right=int((dis & (S == y)).sum()),
        score_corr=float(np.corrcoef(ss, sh)[0, 1]))
    # examples: labelled check-worthy, Haiku yes, Sonnet no (the gap that costs Sonnet accuracy) and the reverse
    idx = np.where(H & ~S & (y == 1))[0]
    rng = np.random.default_rng(0)
    out[name]["ex_haiku_yes_sonnet_no_label1"] = [dict(text=txt[i], sonnet=float(ss[i]), haiku=float(sh[i]))
                                                  for i in rng.choice(idx, min(12, len(idx)), replace=False)]
    idx = np.where(H & ~S & (y == 0))[0]
    out[name]["ex_haiku_yes_sonnet_no_label0"] = [dict(text=txt[i], sonnet=float(ss[i]), haiku=float(sh[i]))
                                                  for i in rng.choice(idx, min(8, len(idx)), replace=False)]

os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results", "analysis"), exist_ok=True)
with open(os.path.join(os.path.dirname(__file__), "..", "results", "analysis", "why_haiku.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

for name in ("clef", "cb"):
    print(f"\n===== {name}")
    for m, lab in LLMS.items():
        r = out[name][m]
        print(f"{lab:10s} test+ {r['test_pos_rate']:.3f} learn+ {r['learn_pos_rate']:.3f} | AUC test {r['auc_test']:.3f} learn {r['auc_learn']:.3f}"
              f" | @0.5 flag {r['flag_rate_05']:.3f} prec {r['prec_05']:.3f} rec {r['rec_05']:.3f} acc {r['acc_05']:.3f}"
              f" | t_learn {r['t_learn']:.2f} acc {r['acc_t_learn']:.3f} | t_oracle {r['t_oracle']:.2f} acc {r['acc_t_oracle']:.3f}"
              f" | mean s pos {r['mean_score_pos']:.2f} neg {r['mean_score_neg']:.2f} | distinct {r['n_distinct_scores']}")
        print("           label rate by score bin: " + "  ".join(
            f"{b['bin']} n={b['n']} {b['label_rate']:.2f}" if b['label_rate'] is not None else f"{b['bin']} n=0" for b in r["bins"]))
    print("agreement", out[name]["agreement"])


# ---- prior-corrected threshold: the learning set is class-balanced (50% positive), but the corpus it was drawn
# from is not; its positive rate is known without test labels (CLEF 2024 train 5,413/22,501; ClaimBuster 2012 debates
# 2,618 sentences, 26.7%). Choose the all-call threshold on the learning set by prior-weighted accuracy.
PRIOR = {"clef": 5413 / 22501, "cb": 0.2666}


def prior_threshold(s, y, pi):
    w = np.where(y == 1, pi / y.mean(), (1 - pi) / (1 - y.mean()))
    ts = np.linspace(0.02, 0.98, 97)
    return float(ts[np.argmax([np.sum(w * ((s >= t) == y)) for t in ts])])


print("\n===== prior-corrected all-call threshold (learning set reweighted to the source corpus' positive rate)")
for name in ("clef", "cb"):
    test, calib = sets(name)
    for m, lab in LLMS.items():
        s, y, _ = scores(test, FS[m])
        sc, yc, _ = scores(calib, FS[m])
        t = prior_threshold(sc, yc, PRIOR[name])
        out[name][m].update(prior=PRIOR[name], t_prior=t, acc_t_prior=acc(s, y, t))
        print(f"{name:5s} {lab:10s} prior {PRIOR[name]:.3f}  t_prior {t:.2f}  acc {acc(s, y, t):.3f}"
              f"   (t_learn {out[name][m]['t_learn']:.2f} acc {out[name][m]['acc_t_learn']:.3f}; 0.5 acc {out[name][m]['acc_05']:.3f};"
              f" oracle {out[name][m]['t_oracle']:.2f} acc {out[name][m]['acc_t_oracle']:.3f})")
with open(os.path.join(os.path.dirname(__file__), "..", "results", "analysis", "why_haiku.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
