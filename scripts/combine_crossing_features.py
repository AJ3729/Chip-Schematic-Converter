"""Combine binary geometry with original-photograph darkness, honestly.

Geometry alone reached image-grouped CV AUC 0.6645 with precision capped at
0.70. Darkness measured on the ORIGINALS adds a physically independent
channel (degree>=4 AUC 0.6548, control at chance). Independent channels can
add; this measures whether they do, and -- more importantly -- whether the
combination reaches the precision a splitting decision requires. A wrong
split severs a net and corrupts every component on it, so precision near
0.95 is the bar, not AUC.
"""
import csv, json, sys
import numpy as np
sys.path.insert(0,'scripts')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import scipy.stats as ss

def auc(pos,neg):
    r=ss.rankdata(np.concatenate([pos,neg]))
    return (r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))

GEO=["dot_ratio","degree","straightness","n_collinear","angle_min_gap",
     "d_xover_box","d_comp_box","local_ink"]
DARK=["dark_at_site","dark_peak","dark_excess_area"]

g={ (r['image'],r['x'],r['y']): r for r in
    csv.DictReader(open('results/real_crossings/sites_test.csv')) }
d={ (r['image'],r['x'],r['y']): r for r in
    csv.DictReader(open('results/ink_darkness_original/sites_darkness_original.csv')) }
keys=sorted(set(g)&set(d))
print(f"joined {len(keys)} sites (geometry {len(g)}, darkness {len(d)})")

rows=[]
for k in keys:
    a,b=g[k],d[k]
    r={'image':k[0],'label':int(a['label'])}
    for f in GEO: r[f]=float(a[f])
    for f in DARK: r[f]=float(b[f])
    rows.append(r)
y=np.array([r['label'] for r in rows]); groups=np.array([r['image'] for r in rows])
print(f"must-split {int(y.sum())}  must-union {int((1-y).sum())}\n")

def cv_auc(feats, label):
    X=np.array([[r[f] for f in feats] for r in rows])
    oof=np.full(len(y),np.nan)
    gkf=GroupKFold(n_splits=5)
    for tr,te in gkf.split(X,y,groups):
        sc=StandardScaler().fit(X[tr])
        clf=LogisticRegression(max_iter=3000,class_weight='balanced')
        clf.fit(sc.transform(X[tr]),y[tr])
        oof[te]=clf.predict_proba(sc.transform(X[te]))[:,1]
    a=auc(oof[y==1],oof[y==0])
    print(f"  {label:34s} CV AUC {a:.4f}")
    return oof,a

print("image-grouped 5-fold CV (no fold scores an image it fit on):")
oof_g,a_g=cv_auc(GEO,"geometry only (binary mask)")
oof_d,a_d=cv_auc(DARK,"darkness only (original photo)")
oof_b,a_b=cv_auc(GEO+DARK,"geometry + darkness")

print(f"\n  darkness adds {a_b-a_g:+.4f} over geometry alone")

print(f"\nOperating points for the combined model. A wrong split severs a net,")
print(f"so precision is the constraint, not AUC:\n")
print(f"  {'thr':>5s} {'split_recall':>13s} {'split_precision':>16s} {'wrong_splits':>13s}")
for thr in (0.5,0.6,0.7,0.8,0.9,0.95):
    tp=int(((oof_b>=thr)&(y==1)).sum()); fp=int(((oof_b>=thr)&(y==0)).sum())
    fn=int(((oof_b<thr)&(y==1)).sum())
    print(f"  {thr:5.2f} {tp/max(tp+fn,1):13.3f} {tp/max(tp+fp,1):16.3f} {fp:13d}")

# degree>=4 subset, where darkness is strongest
deg=np.array([r['degree'] for r in rows])
m=deg>=4
if m.sum()>50:
    print(f"\ndegree>=4 subset ({int(m.sum())} sites, {int(y[m].sum())} split):")
    for nm,o in (("geometry",oof_g),("darkness",oof_d),("combined",oof_b)):
        print(f"  {nm:10s} AUC {auc(o[m&(y==1)],o[m&(y==0)]):.4f}")
    print(f"\n  combined, degree>=4 operating points:")
    for thr in (0.6,0.7,0.8,0.9):
        tp=int(((oof_b>=thr)&(y==1)&m).sum()); fp=int(((oof_b>=thr)&(y==0)&m).sum())
        fn=int(((oof_b<thr)&(y==1)&m).sum())
        print(f"    thr {thr:.1f} recall {tp/max(tp+fn,1):.3f} "
              f"precision {tp/max(tp+fp,1):.3f} wrong_splits {fp}")

json.dump({"n_sites":len(rows),"cv_auc_geometry":round(a_g,4),
           "cv_auc_darkness":round(a_d,4),"cv_auc_combined":round(a_b,4),
           "darkness_gain":round(a_b-a_g,4)},
          open('results/ink_darkness_original/combined_fit.json','w'), indent=2)
print(f"\nwrote results/ink_darkness_original/combined_fit.json")
