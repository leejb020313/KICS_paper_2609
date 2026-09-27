"""Check that every result number printed in the paper PDF matches results/*.json.

Usage: python scripts/verify_paper_numbers.py <paper.pdf>
"""
import json
import os
import re
import sys

import pymupdf

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
R = json.load(open(os.path.join(RESULTS, "final_results.json"), encoding="utf-8"))
S = json.load(open(os.path.join(RESULTS, "streaming_results.json"), encoding="utf-8"))
text = re.sub(r"\s+", " ", "".join(p.get_text() for p in pymupdf.open(sys.argv[1])))
acc = lambda d, n: R[d]["metrics"][n]["acc"][0]
m = lambda d, n, k: R[d]["metrics"][n][k][0]
cal = lambda d, n: R[d]["calib_curve"][n][0]
pv = lambda d, n: R[d]["contrasts"][n]["mcnemar_p"]

checks = []
for d in ("clef", "cb"):
    for n in ["gemma_raw", "gemma_nnppi_sel", "svm", "sonnet_raw", "sonnet_thr", "sonnet_nnppi", "replace_0.5", "fuse_0.5", "fuse_1.0"]:
        checks.append((f"{d} table {n}", f"{acc(d, n):.3f}"))
    checks += [(f"{d} recall sonnet_raw", f"{m(d, 'sonnet_raw', 'rec1'):.2f}"), (f"{d} prec sonnet_raw", f"{m(d, 'sonnet_raw', 'prec1'):.2f}"),
               (f"{d} recall sonnet_thr", f"{m(d, 'sonnet_thr', 'rec1'):.2f}"),
               (f"{d} NN-PPI wF1 (reproduction)", f"{m(d, 'gemma_nnppi_sel', 'wf1'):.3f}"),
               (f"{d} p fuse50 vs sonnet+nnppi", f"{pv(d, 'fuse_0.5_vs_sonnet_nnppi'):.2f}"),
               (f"{d} diff fuse50 - sonnet+nnppi", f"{100 * (acc(d, 'fuse_0.5') - acc(d, 'sonnet_nnppi')):.1f}%p"),
               (f"{d} calib fuse_0.5", f"{cal(d, 'fuse_0.5'):.3f}"), (f"{d} calib sonnet_thr_oof", f"{cal(d, 'sonnet_thr_oof'):.3f}"),
               (f"{d} p svm vs nnppi", f"p={pv(d, 'svm_vs_gemma_nnppi_sel'):.2f}"), (f"{d} k_sel", f"k={R[d]['k_sel']}"),
               (f"{d} stream acc@50", f"{S[d]['acc']['stream_global_0.5'][0]:.3f}"),
               (f"{d} stream rate@50", f"{100 * S[d]['test_call_rate']['0.5'][0]:.0f}%")]
checks += [("cb p fuse50 vs sonnet", f"p={pv('cb', 'fuse_0.5_vs_sonnet_thr'):.3f}"),
           ("clef p fuse50 vs sonnet", f"p={pv('clef', 'fuse_0.5_vs_sonnet_thr'):.2f}"),
           ("clef diff pp", f"{100 * (acc('clef', 'fuse_0.5') - acc('clef', 'sonnet_thr')):.1f}%p"),
           ("cb replace p max", f"p≤{max(pv('cb', f'fuse_{b}_vs_replace_{b}') for b in (0.3, 0.4, 0.5)):.3f}"),
           ("calib gap <=0.7pp", "0.7%p"),
           ("calib gain after 50%", f"{100 * max(cal(d, f'fuse_{b}') - cal(d, 'fuse_0.5') for d in ('clef', 'cb') for b in (.6, .7, .8, .9, 1.0)):.1f}%p"),
           ("cb flips replace broke", f"{R['cb']['flips_confident_half_seed0']['replace']['broke']}건"),
           ("cb flips replace fixed", f"{R['cb']['flips_confident_half_seed0']['replace']['fixed']}건"),
           ("cb flips fuse broke", f"{R['cb']['flips_confident_half_seed0']['fuse']['broke']}건")]
gap = max(cal(d, "sonnet_thr_oof") - cal(d, "fuse_0.5") for d in ("clef", "cb"))
bad = 0
for name, s in checks:
    ok = s in text or s.replace(" ", "") in text.replace(" ", "")
    bad += not ok
    print(f"{'OK ' if ok else 'MISSING'}  {name:26s} {s}")
print(f"\nmax calib gap frontier-fused@50% = {100 * gap:.2f}pp (text claims within 0.7pp)")
print("ALL MATCH" if bad == 0 and gap <= 0.007 + 1e-9 else f"{bad} mismatches")
