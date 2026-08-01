"""Split vs weld decomposition for the recoverable-12 against the hard-core-81."""
import sys, csv, json, statistics
sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
from pathlib import Path
from schematic2netlist.benchmark import align_components, canonicalize_terminals
from schematic2netlist.config import load_config
from schematic2netlist.gt import gt_to_components, load_gt
from schematic2netlist.metrics import _terminal_pairs
from schematic2netlist.pipeline import run_pipeline
from vlm_task import load_detections

def load(p): return {r['image']:r for r in csv.DictReader(open(p))}
pipe=load('results/benchmark_1024_final/seed0/per_image.csv')
cl=load('results/vlm/claude_b/scored/per_image.csv')
gp=load('results/vlm/openai_b/scored/per_image.csv')
ok=lambda d,i: d[i]['strict_success']=='True'
imgs=[i for i in pipe if i in cl and i in gp]
groups={'recoverable-12':[i for i in imgs if not ok(pipe,i) and ok(cl,i) and ok(gp,i)],
        'hard-core-81'  :[i for i in imgs if not ok(pipe,i) and not ok(cl,i) and not ok(gp,i)]}
cfg=load_config(None); idir=Path(cfg['preprocess']['images_dir'])

out={}
for lab,S in groups.items():
    miss=extra=0; rows=[]
    for nm in S:
        stem=Path(nm).stem
        gt=load_gt(f"{cfg['benchmark']['gt_dir']}/{stem}.json")
        gc=gt_to_components(gt); by={c['id']:c for c in gt['components']}
        for c in gc: c['bbox']=by[c['id']]['bbox']
        dets=load_detections(stem,cfg)
        res=run_pipeline(str(idir/nm),cfg,detections=dets)
        pred=[{'id':c['id'],'class':c['class'],'nets':list(c.get('node_names',[])),
               'bbox':[res['detections'][c['id']]['x'],res['detections'][c['id']]['y'],
                       res['detections'][c['id']]['width'],res['detections'][c['id']]['height']]}
              for c in res['components']]
        p,g,_=align_components(pred,gc)
        pp,gg=_terminal_pairs(canonicalize_terminals(p)),_terminal_pairs(canonicalize_terminals(g))
        m,e=len(gg-pp),len(pp-gg)
        miss+=m; extra+=e
        rows.append((nm,m,e))
    out[lab]=(miss,extra,rows)
    print(f'{lab:<16} missing(split) {miss:>5}   extra(weld) {extra:>5}   '
          f'ratio miss:extra = {miss/max(1,extra):.2f}')
    n_split=sum(1 for _,m,e in rows if m>e); n_weld=sum(1 for _,m,e in rows if e>m)
    print(f'{"":16} images split-dominant {n_split}/{len(rows)}, weld-dominant {n_weld}/{len(rows)}')
json.dump({k:[v[0],v[1]] for k,v in out.items()}, open(f'{sys.argv[1]}/twelve.json','w'))
