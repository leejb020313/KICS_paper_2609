"""Check every result number printed in the draft PDF against a freshly produced final_results.json."""
import json
import re
import sys

import pymupdf

R = json.load(open(sys.argv[1], encoding="utf-8"))
text = re.sub(r"\s+", " ", "".join(p.get_text() for p in pymupdf.open(sys.argv[2])))
acc = lambda d, n: R[d]["metrics"][n]["acc"][0]
f1 = lambda d, n: R[d]["metrics"][n]["f1c1"][0]
cal = lambda d, n: R[d]["calib_curve"][n][0]
pv = lambda d, n: R[d]["contrasts"][n]["mcnemar_p"]

checks = []
for d in ("clef", "cb"):
    for n in ["gemma_raw", "gemma_nnppi_sel", "svm", "sonnet_raw", "sonnet_thr", "replace_0.5", "fuse_0.5", "fuse_1.0"]:
        checks.append((f"{d} table {n}", f"{acc(d, n):.3f}"))
    checks += [(f"{d} F1c1 sonnet_raw", f"{f1(d, 'sonnet_raw'):.3f}"), (f"{d} F1c1 sonnet_thr", f"{f1(d, 'sonnet_thr'):.3f}"),
               (f"{d} calib fuse_0.5", f"{cal(d, 'fuse_0.5'):.3f}"), (f"{d} calib sonnet_thr_oof", f"{cal(d, 'sonnet_thr_oof'):.3f}"),
               (f"{d} p svm vs nnppi", f"p={pv(d, 'svm_vs_gemma_nnppi_sel'):.2f}"), (f"{d} k_sel", f"k={R[d]['k_sel']}")]
checks += [("cb p fuse50 vs sonnet", f"p={pv('cb', 'fuse_0.5_vs_sonnet_thr'):.3f}"),
           ("clef p fuse50 vs sonnet", f"p={pv('clef', 'fuse_0.5_vs_sonnet_thr'):.2f}"),
           ("clef diff pp", f"{100 * (acc('clef', 'fuse_0.5') - acc('clef', 'sonnet_thr')):.1f}%p"),
           ("cb replace p max", f"p≤{max(pv('cb', f'fuse_{b}_vs_replace_{b}') for b in (0.3, 0.4, 0.5)):.3f}"),
           ("calib gap <=0.7pp", "0.7%p"), ("cb fuse50->100 pp", f"{100*(acc('cb','fuse_1.0')-acc('cb','fuse_0.5')):.1f}%p")]
import os
IMP = json.load(open(os.path.join(os.path.dirname(sys.argv[1]), "improve_results.json"), encoding="utf-8"))
for d in ("clef", "cb"):
    checks.append((f"{d} stream acc@50", f"{IMP[d]['acc']['stream_global_0.5'][0]:.3f}"))
    checks.append((f"{d} stream rate@50", f"{100*IMP[d]['test_call_rate']['0.5'][0]:.0f}%"))
gap = max(cal(d, "sonnet_thr_oof") - cal(d, "fuse_0.5") for d in ("clef", "cb"))
bad = 0
for name, s in checks:
    ok = s in text
    bad += not ok
    print(f"{'OK ' if ok else 'MISSING'}  {name:26s} {s}")
print(f"\nmax calib gap frontier-fused@50% = {100 * gap:.2f}pp (text claims within 0.7pp)")
print("ALL MATCH" if bad == 0 and gap <= 0.007 + 1e-9 else f"{bad} mismatches")
