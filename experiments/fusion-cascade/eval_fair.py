import json,re,glob,os
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.stats import binomtest
from sentence_transformers import SentenceTransformer
ev=json.load(open('frontier_eval_set.json',encoding='utf-8')); cs=json.load(open('frontier_calib_set.json',encoding='utf-8'))
fs={}
for f in glob.glob('batches/*.out'):
    k=os.path.basename(f).split('_')[0]; t=open(f,encoding='utf-8',errors='ignore').read()
    fs.setdefault(k,{}).update({int(a):float(b) for a,b in json.loads(re.search(r'\{.*\}',t,re.S).group(0)).items()})
emb=SentenceTransformer('all-MiniLM-L6-v2')
E=lambda x: np.asarray(emb.encode(x,normalize_embeddings=True,show_progress_bar=False))
def nnppi(s_cal,y_cal,Xc,s_te,Xt,k=10):
    sim=Xt@Xc.T; idx=np.argsort(-sim,1)[:,:k]; r=(y_cal-s_cal)[idx].mean(1); return np.clip(s_te+r,0,1)
def best_thr(s,y):
    ts=np.linspace(0.05,0.95,91); return ts[np.argmax([f1_score(y,(s>=t).astype(int),average='weighted') for t in ts])]
def m(y,p): return f"acc={np.mean(p==y):.3f} wF1={f1_score(y,p,average='weighted'):.3f} F1cls1={f1_score(y,p):.3f}"
for k,ck in [('clef','clefcal'),('cb','cbcal')]:
    have=sorted(fs[ck]); C=[cs[ck][i] for i in have]; yc=np.array([r['label'] for r in C]); sc=np.array([fs[ck][i] for i in have])
    T=ev[k]; y=np.array([r['label'] for r in T]); st=np.array([fs[k][i] for i in range(len(T))])
    Xc=E([r['text'] for r in C]); Xt=E([r['text'] for r in T])
    # hold out 30% of calib for threshold selection (no test peeking)
    rng=np.random.default_rng(0); perm=rng.permutation(len(C)); tune=perm[:int(.3*len(C))]; pool=perm[int(.3*len(C)):]
    print(f"\n===== {k}: test n={len(T)}, calib used n={len(C)} (frontier-scored subset)")
    P={}
    P['Sonnet raw (thr .5)']=(st>=.5).astype(int)
    t=best_thr(sc[tune],yc[tune]); P[f'Sonnet + thr tuned on calib ({t:.2f})']=(st>=t).astype(int)
    tn=nnppi(sc[pool],yc[pool],Xc[pool],sc[tune],Xc[tune]); t2=best_thr(tn,yc[tune])
    P[f'Sonnet + NN-PPI k=10 (thr {t2:.2f})']=(nnppi(sc[pool],yc[pool],Xc[pool],st,Xt)>=t2).astype(int)
    svm=SVC(class_weight='balanced',random_state=0).fit(Xc,yc); P['Emb-SVM (same calib, no LLM)']=svm.predict(Xt)
    # stack: SVM decision + Sonnet score, fit on calib via cross-fitting
    from sklearn.model_selection import cross_val_predict
    dcal=cross_val_predict(SVC(class_weight='balanced',random_state=0),Xc,yc,cv=5,method='decision_function')
    lr=LogisticRegression(class_weight='balanced').fit(np.c_[dcal,sc],yc)
    P['Emb-SVM + Sonnet stacked']=lr.predict(np.c_[svm.decision_function(Xt),st])
    for n,p in P.items(): print(f"  {n:38s} {m(y,p)}")
    a=P['Emb-SVM (same calib, no LLM)']==y
    for n in list(P)[:3]:
        b=P[n]==y; n01=int((a&~b).sum()); n10=int((~a&b).sum())
        print(f"    SVM vs [{n}]: SVM-only-right={n01} other-only-right={n10} McNemar p={binomtest(n01,n01+n10).pvalue:.3f}")
