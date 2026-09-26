"""Diagnostic: cascade accuracy curve on calib (out-of-fold) vs test, seed 0, to see why the calib rule picks rho=1."""
import glob, json, os, re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.svm import SVC
ev = json.load(open('frontier_eval_set.json', encoding='utf-8')); cs = json.load(open('frontier_calib_set.json', encoding='utf-8'))
fs = {}
for f in glob.glob('batches/*.out'):
    k = os.path.basename(f).split('_')[0]
    fs.setdefault(k, {}).update({int(a): float(b) for a, b in json.loads(re.search(r'\{.*\}', open(f, encoding='utf-8').read(), re.S).group(0)).items()})
emb = SentenceTransformer('all-MiniLM-L6-v2'); E = lambda x: np.asarray(emb.encode(x, normalize_embeddings=True, show_progress_bar=False))
B = [0, .1, .2, .3, .4, .5, .7, 1.0]
def thr_fit(s, y):
    ts = np.linspace(.02, .98, 97); return ts[np.argmax([np.mean((s >= t) == y) for t in ts])]
for k, ck in [('clef', 'clefcal'), ('cb', 'cbcal')]:
    have = sorted(fs[ck]); C = [cs[ck][i] for i in have]; yc = np.array([r['label'] for r in C]); sc = np.array([fs[ck][i] for i in have]); Xc = E([r['text'] for r in C])
    idx = np.random.default_rng(0).permutation(len(C))[:int(.8 * len(C))]; X, y, s = Xc[idx], yc[idx], sc[idx]
    d = cross_val_predict(SVC(class_weight='balanced', random_state=0), X, y, cv=5, method='decision_function')
    pf = cross_val_predict(LogisticRegression(class_weight='balanced'), np.c_[d, s], y, cv=5)
    # out-of-fold threshold for the frontier reference
    son_oof = np.zeros(len(y), int)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        son_oof[te] = (s[te] >= thr_fit(s[tr], y[tr])).astype(int)
    print(f"\n{k}: calib n={len(y)}  SVM-OOF acc={np.mean((d>0)==y):.3f}  Sonnet+thr in-sample={np.mean((s>=thr_fit(s,y))==y):.3f}  Sonnet+thr OOF={np.mean(son_oof==y):.3f}")
    o = np.argsort(np.abs(d)); p0 = (d > 0).astype(int)
    print("  calib OOF fused cascade:", "  ".join(f"{int(b*100)}%={np.mean(np.where(np.isin(np.arange(len(y)), o[:int(round(b*len(y)))]), pf, p0)==y):.3f}" for b in B))
