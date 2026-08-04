"""Is a self-shorted component a high-precision weld detector?

Only 0.60% of GT components with >=2 terminals put two pins on one net. So a
PREDICTED component whose pins all land on the same node is, on that prior,
almost certainly a weld -- and unlike a skeleton site, its location is a
detector box accurate to ~2 px. Measure the precision of that signal.
"""
import argparse
import sys, csv, json
from pathlib import Path
from collections import Counter
import numpy as np
from schematic2netlist.config import load_config
from schematic2netlist.detect import load_cached_detections
from schematic2netlist.gt import load_gt, gt_to_components
from schematic2netlist.pipeline import run_pipeline
from schematic2netlist.benchmark import align_components, canonicalize_terminals
from schematic2netlist.classes import canonical_class

from schematic2netlist.splits import load_split

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--split", default="val",
                help="exploratory measurement, so it reads val by default and a "
                     "prior measured here cannot leak into a reported number")
ap.add_argument("--limit", type=int, default=None)
args = ap.parse_args()

cfg = load_config(None)
names = load_split(args.split)
if args.limit:
    names = names[: args.limit]
from schematic2netlist.frames import resolve_and_check
images_dir = resolve_and_check(None, names, cfg)

n_comp = 0; n_pred_short = 0; n_pred_short_and_gt_short = 0
by_class = Counter(); tot_by_class = Counter()
per_image = []
for i, nm in enumerate(names, 1):
    stem = Path(nm).stem
    gt = load_gt(f"{cfg['benchmark']['gt_dir']}/{stem}.json")
    gcomps = gt_to_components(gt)
    by = {c['id']: c for c in gt['components']}
    for c in gcomps: c['bbox'] = by[c['id']]['bbox']
    dets = load_cached_detections(f"{cfg['detect']['cache_dir']}/{stem}.json",
                                 min_confidence=cfg['detect'].get('confidence'))
    res = run_pipeline(images_dir / nm, cfg, detections=dets)
    pred = [{'id': c['id'], 'class': c['class'],
             'nets': list(c.get('node_names', [])),
             'bbox': [res['detections'][c['id']]['x'], res['detections'][c['id']]['y'],
                      res['detections'][c['id']]['width'], res['detections'][c['id']]['height']]}
            for c in res['components']]
    p, g, _ = align_components(pred, gcomps)
    pc, gc = canonicalize_terminals(p), canonicalize_terminals(g)
    gmap = {c['id']: c for c in gc}
    img_short = 0
    for c in pc:
        nets = [n for n in c['nets'] if n is not None]
        if len(nets) < 2: continue
        gcm = gmap.get(c['id'])
        if gcm is None: continue
        gnets = [n for n in gcm['nets'] if n is not None]
        if len(gnets) < 2: continue
        cls = canonical_class(c['class'])
        n_comp += 1; tot_by_class[cls] += 1
        pred_short = len(set(nets)) < len(nets)
        gt_short = len(set(gnets)) < len(gnets)
        if pred_short:
            n_pred_short += 1; img_short += 1; by_class[cls] += 1
            if gt_short: n_pred_short_and_gt_short += 1
    per_image.append({'image': nm, 'n_self_shorted': img_short})
    if i % 25 == 0: print(f'  [{i}/{len(names)}]', flush=True)

print(f"\ncomponents with >=2 terminals scored: {n_comp}")
print(f"PREDICTED self-shorted (all pins on one node): {n_pred_short} = {n_pred_short/max(n_comp,1):.2%}")
print(f"  of those, GT agrees it is genuinely shorted : {n_pred_short_and_gt_short}")
print(f"  => WELD-DETECTION PRECISION = {1 - n_pred_short_and_gt_short/max(n_pred_short,1):.4f}")
print(f"\nGT baseline rate of genuine shorts: 0.60%")
print(f"\nby class:")
for cls, n in by_class.most_common(10):
    print(f"  {cls:20s} {n:4d}/{tot_by_class[cls]:4d} = {n/tot_by_class[cls]:6.1%}")
out = Path('results/real_crossings'); out.mkdir(parents=True, exist_ok=True)
(out / 'self_short_detector.json').write_text(json.dumps({
  'n_components_scored': n_comp, 'n_pred_self_shorted': n_pred_short,
  'n_also_gt_shorted': n_pred_short_and_gt_short,
  'weld_detection_precision': round(1 - n_pred_short_and_gt_short/max(n_pred_short,1), 4),
  'gt_genuine_short_rate': 0.006,
  'by_class': {k: {'n_shorted': v, 'n_total': tot_by_class[k]} for k, v in by_class.items()},
  'per_image': per_image,
}, indent=2) + "\n")
print(f"\nwrote {out}/self_short_detector.json")
