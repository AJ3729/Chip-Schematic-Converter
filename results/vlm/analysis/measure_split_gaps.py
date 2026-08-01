"""For every GT net the pipeline SPLIT, how wide is the gap and what sits in it?

A split is a net whose terminals landed on several predicted nodes. If the gap
between the fragments is small and empty, the fix is a wider bridge. If a
component or text box sits in it, the fix is masking. If the gap is large, the
drafter's conductor was never recovered at all.
"""
import sys, csv, json
import numpy as np, cv2
sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
from pathlib import Path
from collections import defaultdict, Counter
from schematic2netlist.benchmark import align_components, canonicalize_terminals
from schematic2netlist.classes import canonical_class, class_terminals
from schematic2netlist.config import load_config
from schematic2netlist.gt import gt_to_components, load_gt
from schematic2netlist.nodes import bbox_xyxy
from schematic2netlist.pipeline import run_pipeline
from vlm_task import load_detections

def load(p): return {r['image']:r for r in csv.DictReader(open(p))}
pipe=load('results/benchmark_1024_final/seed0/per_image.csv')
cl=load('results/vlm/claude_b/scored/per_image.csv'); gp=load('results/vlm/openai_b/scored/per_image.csv')
ok=lambda d,i: d[i]['strict_success']=='True'
imgs=[i for i in pipe if i in cl and i in gp]
S12=[i for i in imgs if not ok(pipe,i) and ok(cl,i) and ok(gp,i)]
cfg=load_config(None); idir=Path(cfg['preprocess']['images_dir'])

gaps=[]; inbox=Counter(); rows=[]
for nm in S12:
    stem=Path(nm).stem
    gt=load_gt(f"{cfg['benchmark']['gt_dir']}/{stem}.json")
    gc=gt_to_components(gt); by={c['id']:c for c in gt['components']}
    for c in gc: c['bbox']=by[c['id']]['bbox']
    dets=load_detections(stem,cfg)
    res=run_pipeline(str(idir/nm),cfg,detections=dets)
    node_map=res['node_map']
    pred=[{'id':c['id'],'class':c['class'],'nets':list(c.get('node_names',[])),
           'bbox':[res['detections'][c['id']]['x'],res['detections'][c['id']]['y'],
                   res['detections'][c['id']]['width'],res['detections'][c['id']]['height']]}
          for c in res['components']]
    p,g,_=align_components(pred,gc)
    pc,gcn=canonicalize_terminals(p),canonicalize_terminals(g)
    name_to_id={}
    for c in res['components']:
        for n_,nn_ in zip(c.get('nodes',[]),c.get('node_names',[])):
            if n_ is not None and nn_ is not None: name_to_id[nn_]=int(n_)
    # GT net -> set of predicted node ids carrying its terminals
    gt2nodes=defaultdict(set)
    pof={(c['id'],k):n for c in pc for k,n in enumerate(c['nets'])}
    for c in gcn:
        for k,gn in enumerate(c['nets']):
            pn=pof.get((c['id'],k))
            if gn and pn and pn in name_to_id: gt2nodes[gn].add(name_to_id[pn])
    boxes=[bbox_xyxy(d) for d in res['detections']]
    for gn,nodes in gt2nodes.items():
        if len(nodes)<2: continue
        pts={n:np.argwhere(node_map==n) for n in nodes}
        pts={n:v for n,v in pts.items() if len(v)}
        ns=sorted(pts)
        for a in range(len(ns)):
            for b in range(a+1,len(ns)):
                A,B=pts[ns[a]],pts[ns[b]]
                if len(A)>4000: A=A[::len(A)//4000]
                if len(B)>4000: B=B[::len(B)//4000]
                d=np.sqrt(((A[:,None,:]-B[None,:,:])**2).sum(-1))
                k=np.unravel_index(d.argmin(),d.shape); gap=float(d[k])
                my,mx=(A[k[0]]+B[k[1]])/2
                hit=any(x1<=mx<=x2 and y1<=my<=y2 for x1,y1,x2,y2 in boxes)
                gaps.append(gap); inbox[hit]+=1
                rows.append({'image':nm,'gt_net':gn,'gap_px':round(gap,1),'in_component_box':hit})
gaps=np.array(gaps)
print(f'split fragment pairs in the 12: {len(gaps)}')
print(f'  gap px  median {np.median(gaps):.1f}   p25 {np.percentile(gaps,25):.1f}   '
      f'p75 {np.percentile(gaps,75):.1f}   max {gaps.max():.1f}')
for t in (7,10,15,20,30,60):
    print(f'  gap <= {t:>3} px : {(gaps<=t).sum():>4} / {len(gaps)}  ({(gaps<=t).mean():.0%})')
print(f'\n  midpoint inside a detected component box: {inbox[True]} / {sum(inbox.values())} '
      f'({inbox[True]/max(1,sum(inbox.values())):.0%})')
print(f'\n  current wires.bridge_span = {cfg["wires"]["bridge_span"]}')
csv.DictWriter(open(f'{sys.argv[1]}/split_gaps.csv','w',newline=''),
               fieldnames=list(rows[0])).writeheader()
with open(f'{sys.argv[1]}/split_gaps.csv','a',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writerows(rows)
