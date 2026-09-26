import json
import numpy as np
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
import sys; sys.path.insert(0, '../..')
from src.nnppi.calibration import NNPPIConfig, nn_ppi_apply
ev=json.load(open('frontier_eval_set.json',encoding='utf-8'))
def load(f): return [json.loads(l) for l in open('../../results/'+f+'.jsonl',encoding='utf-8') if l.strip()]
emb=SentenceTransformer('all-MiniLM-L6-v2'); E=lambda x: np.asarray(emb.encode(x,normalize_embeddings=True,show_progress_bar=False))
for k,cal,test in [('clef','clef_calib_scores','clef_test_scores'),('cb','claimbuster_calib_scores','claimbuster_test_scores')]:
    C=[r for r in load(cal) if r.get('parse_ok')]; sc=np.array([r['confidence_score'] for r in C],float); yc=np.array([r['label'] for r in C])
    gem={str(r['Sentence_id']):r for r in load(test)}
    T=ev[k]; y=np.array([r['label'] for r in T])
    st=np.array([gem[r['Sentence_id']]['confidence_score'] if gem[r['Sentence_id']].get('parse_ok') else 0.0 for r in T],float)
    Xc=E([r['text'] for r in C]); Xt=E([r['text'] for r in T])
    print(f"\n===== {k} same {len(T)} test items; Gemma calib n={len(C)} (full)")
    def rep(n,p): print(f"  {n:34s} acc={np.mean(p==y):.3f} wF1={f1_score(y,p,average='weighted'):.3f} F1cls1={f1_score(y,p):.3f}")
    rep('Gemma 4B raw',(st>=.5).astype(int))
    for kk in [3,5,10]:
        th=nn_ppi_apply(st,Xt,sc,Xc,yc,NNPPIConfig(k=kk)).theta
        rep(f'Gemma 4B + NN-PPI (paper) k={kk}',(th>=.5).astype(int))
