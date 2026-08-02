"""Is the recoverable-split population characterised by faint or sparse ink?

If the pipeline loses long conductors on these circuits, the candidate causes
are (a) binarization dropping faint strokes, (b) the blob filter deleting real
segments, (c) masking erasing them. All three are self-inflicted and all three
would show up as less surviving ink relative to the original photograph.
"""
import sys, csv, json
sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
from pathlib import Path
import numpy as np, cv2, statistics as st
from schematic2netlist.config import load_config
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
for lab,S in G.items():
    frac_ink=[]; frac_wire=[]; kept=[]; ncomp=[]
    for nm in S:
        stem=Path(nm).stem
        gray=cv2.imread(str(idir/nm), cv2.IMREAD_GRAYSCALE)
        ink=(gray<200).mean()                      # ink in the cleaned frame
        dets=load_detections(stem,cfg)
        res=run_pipeline(str(idir/nm),cfg,detections=dets)
        w=res['clean_wires']>0
        frac_ink.append(ink); frac_wire.append(w.mean())
        kept.append(w.mean()/max(ink,1e-9))        # wire mask / total ink
        n,_=cv2.connectedComponents(w.astype(np.uint8),connectivity=8)
        ncomp.append(n-1)
    print(f'{lab:<16} n={len(S):>3}  ink {st.mean(frac_ink):.4f}  wiremask {st.mean(frac_wire):.4f}  '
          f'wire/ink {st.mean(kept):.3f}  wire-mask components {st.mean(ncomp):>5.1f}')
