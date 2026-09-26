import json,re,glob,time
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from scipy.stats import binomtest
from sentence_transformers import SentenceTransformer
ev=json.load(open('frontier_eval_set.json',encoding='utf-8'))
fs={}
for f in glob.glob('batches/*.out'):
    import os; k=os.path.basename(f).split('_')[0]
    t=open(f,encoding='utf-8',errors='ignore').read()
    m=re.search(r'\{.*\}',t,re.S)
    d=json.loads(m.group(0)); fs.setdefault(k,{}).update({int(a):float(b) for a,b in d.items()})
def load(f): return [json.loads(l) for l in open('../../results/'+f+'.jsonl',encoding='utf-8') if l.strip()]
emb=SentenceTransformer('all-MiniLM-L6-v2')
def E(x): return np.asarray(emb.encode(x,normalize_embeddings=True,show_progress_bar=False))
for k,cal,test in [('clef','clef_calib_scores','clef_test_scores'),('cb','claimbuster_calib_scores','claimbuster_test_scores')]:
    C=[r for r in load(cal) if r.get('parse_ok')]
    gem={str(r['Sentence_id']):r.get('confidence_score') for r in load(test)}
    items=ev[k]; y=np.array([r['label'] for r in items])
    t0=time.time(); Xc=E([r['text'] for r in C]); svm=SVC(class_weight='balanced',random_state=0).fit(Xc,[r['label'] for r in C]); t_train=time.time()-t0
    t0=time.time(); p_svm=svm.predict(E([r['text'] for r in items])); t_inf=(time.time()-t0)/len(items)
    miss=[i for i in range(len(items)) if i not in fs[k]]
    p_son=np.array([int(fs[k].get(i,0)>=0.5) for i in range(len(items))])
    g=np.array([ (gem[r['Sentence_id']] if gem[r['Sentence_id']] is not None else 0) for r in items]); p_gem=(g>=0.5).astype(int)
    print(f"\n===== {k}  n={len(items)}  pos-rate={y.mean():.3f}  sonnet-missing={len(miss)}  calib n={len(C)}")
    for name,p in [('Gemma3-4B raw (few-shot)',p_gem),('Emb-SVM (MiniLM+calib, CPU)',p_svm),('Claude Sonnet 5 (frontier)',p_son)]:
        print(f"  {name:30s} acc={np.mean(p==y):.3f}  wF1={f1_score(y,p,average='weighted'):.3f}  F1cls1={f1_score(y,p):.3f}  F1cls0={f1_score(1-y,1-p):.3f}")
    a=(p_svm==y); b=(p_son==y); n01=int((a&~b).sum()); n10=int((~a&b).sum())
    print(f"  SVM right/Sonnet wrong={n01}  Sonnet right/SVM wrong={n10}  McNemar p={binomtest(n01,n01+n10).pvalue:.4f}")
    rng=np.random.default_rng(0); d=[]
    for _ in range(2000):
        ix=rng.integers(0,len(y),len(y)); d.append(np.mean(a[ix])-np.mean(b[ix]))
    print(f"  acc(SVM)-acc(Sonnet) = {np.mean(a)-np.mean(b):+.3f}  bootstrap95%CI [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]")
    print(f"  SVM train {t_train:.1f}s (incl. embedding {len(C)} calib), inference {1000*t_inf:.2f} ms/claim on CPU")
