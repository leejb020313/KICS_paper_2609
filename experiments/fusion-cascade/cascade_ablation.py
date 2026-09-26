import json,re,glob,os
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from scipy.stats import binomtest
from sentence_transformers import SentenceTransformer
ev=json.load(open('frontier_eval_set.json',encoding='utf-8')); cs=json.load(open('frontier_calib_set.json',encoding='utf-8'))
fs={}
for f in glob.glob('batches/*.out'):
    k=os.path.basename(f).split('_')[0]; t=open(f,encoding='utf-8',errors='ignore').read()
    fs.setdefault(k,{}).update({int(a):float(b) for a,b in json.loads(re.search(r'\{.*\}',t,re.S).group(0)).items()})
emb=SentenceTransformer('all-MiniLM-L6-v2'); E=lambda x: np.asarray(emb.encode(x,normalize_embeddings=True,show_progress_bar=False))
def best_thr(s,y):
    ts=np.linspace(0.05,0.95,91); return ts[np.argmax([np.mean((s>=t)==y) for t in ts])]
for k,ck in [('clef','clefcal'),('cb','cbcal')]:
    have=sorted(fs[ck]); C=[cs[ck][i] for i in have]; yc=np.array([r['label'] for r in C]); sc=np.array([fs[ck][i] for i in have])
    T=ev[k]; y=np.array([r['label'] for r in T]); st=np.array([fs[k][i] for i in range(len(T))])
    Xc=E([r['text'] for r in C]); Xt=E([r['text'] for r in T])
    svm=SVC(class_weight='balanced',random_state=0).fit(Xc,yc); dt=svm.decision_function(Xt)
    dcal=cross_val_predict(SVC(class_weight='balanced',random_state=0),Xc,yc,cv=5,method='decision_function')
    lr=LogisticRegression(class_weight='balanced').fit(np.c_[dcal,sc],yc)
    p_svm=(dt>0).astype(int); p_stack=lr.predict(np.c_[dt,st])
    thr=best_thr(sc,yc); p_son=(st>=thr).astype(int)
    order=np.argsort(np.abs(dt))  # most uncertain first
    print(f"\n===== {k} (n={len(y)})   Sonnet+thr(all calls)={np.mean(p_son==y):.3f}  stack(all calls)={np.mean(p_stack==y):.3f}  SVM(0 calls)={np.mean(p_svm==y):.3f}")
    print("  frontier-call-rate   acc    wF1   F1cls1   vs Sonnet+thr(100%): only-casc-right / only-son-right  p")
    for f in [0,.1,.2,.3,.4,.5,.7,1.0]:
        n=int(round(f*len(y))); idx=order[:n]
        p=p_svm.copy(); p[idx]=p_stack[idx]
        q=p_svm.copy(); q[idx]=p_son[idx]
        print(f"  {f:>6.0%}  FUSE acc={np.mean(p==y):.3f} F1c1={f1_score(y,p):.3f}  |  REPLACE acc={np.mean(q==y):.3f} F1c1={f1_score(y,q):.3f}  | fuse-only-right={int(((p==y)&(q!=y)).sum())} replace-only-right={int(((q==y)&(p!=y)).sum())} p={binomtest(int(((p==y)&(q!=y)).sum()),max(1,int(((p==y)!=(q==y)).sum()))).pvalue:.3f}")
        continue
        a=p==y; b=p_son==y; n01=int((a&~b).sum()); n10=int((~a&b).sum()); pv=binomtest(n01,n01+n10).pvalue if n01+n10 else 1
        print(f"  {f:>6.0%}             {np.mean(a):.3f}  {f1_score(y,p,average='weighted'):.3f}  {f1_score(y,p):.3f}    {n01:3d} / {n10:3d}   p={pv:.3f}")
