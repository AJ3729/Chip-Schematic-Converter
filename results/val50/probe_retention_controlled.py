"""Is the low wire/ink retention a defect, or just more components erased?

Component boxes are erased from the wire mask by design. The recoverable
circuits average 18.8 GT components against 9.8 for circuits we solve, so a
lower wire/ink ratio could be entirely that. Recompute retention over pixels
OUTSIDE every detected component box, where erasure cannot explain a loss.
"""
import sys, csv
sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
from pathlib import Path
import numpy as np, cv2, statistics as st
from schematic2netlist.config import load_config
from schematic2netlist.nodes import bbox_xyxy
from schematic2netlist.pipeline import run_pipeline
from vlm_task import load_detections

def load(p): return {r['image']:r for r in csv.DictReader(open(p))}
pipe=load('results/benchmark_1024_final/seed0/per_image.csv')
cl=load('results/vlm/claude_b/scored/per_image.csv'); gp=load('results/vlm/openai_b/scored/per_image.csv')
ok=lambda d,i: d[i]['strict_success']=='True'
imgs=[i for i in pipe if i in cl and i in gp]
G={'recoverable-25':[i for i in imgs if not ok(pipe,i) and (ok(cl,i) or ok(gp,i))],
   'hard-core-81'  :[i for i in imgs if not ok(pipe,i) and not ok(cl,i) and not ok(gp,i)],
   'solved-84'     :[i for i in imgs if ok(pipe,i)]}
cfg=load_config(None); idir=Path(cfg['preprocess']['images_dir'])
print(f"{'group':<16}{'n':>4}{'ncomp':>7}{'wire/ink ALL':>14}{'wire/ink OUTSIDE boxes':>24}")
for lab,S in G.items():
    allr=[]; outr=[]; nc=[]
    for nm in S:
        stem=Path(nm).stem
        gray=cv2.imread(str(idir/nm), cv2.IMREAD_GRAYSCALE)
        dets=load_detections(stem,cfg)
        res=run_pipeline(str(idir/nm),cfg,detections=dets)
        ink=gray<200
        w=res['clean_wires']>0
        outside=np.ones_like(ink)
        for d in dets:
            x1,y1,x2,y2=[int(v) for v in bbox_xyxy(d)]
            outside[max(0,y1):y2, max(0,x1):x2]=False
        allr.append(w.sum()/max(ink.sum(),1))
        outr.append((w&outside).sum()/max((ink&outside).sum(),1))
        nc.append(len(dets))
    print(f'{lab:<16}{len(S):>4}{st.mean(nc):>7.1f}{st.mean(allr):>14.3f}{st.mean(outr):>24.3f}')
