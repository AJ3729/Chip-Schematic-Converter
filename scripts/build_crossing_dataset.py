#!/usr/bin/env python3
"""Self-labeled crossing/junction patches from synthetic routed renders.

The CGHD-trained classifier failed in the pipeline for three measured
reasons: its training strokes look nothing like our masks (domain gap),
its labels encode DRAWN conventions rather than electrical truth, and a
48 px patch cannot see stroke continuation. This generator removes all
three at once by rendering wiring whose net structure WE choose:

- **Layouts are real.** Component boxes come from the published COCO
  annotations of the train/val splits, projected into the 1024 frame —
  the same projection the detector labels use. The verified test split
  is never read.
- **Topology is synthesized, so labels are exact and free.** Terminals
  are partitioned into random nets and routed with the oracle's Lee
  maze router (which lets nets cross freely, as real schematics do).
  A cell where two different nets' paths overlap IS a crossing; a cell
  where one net's own tree branches IS a junction. No annotator, no
  convention, no noise.
- **Rendering mimics the pipeline's wire masks.** Hand-drawn-style
  augmentation (stroke-width jitter, low-frequency wobble, pen-lift
  gaps), junction dots and crossover hops at configurable rates, and
  component interiors erased exactly as the pipeline erases them
  (pad 0). Patches are cropped with the SAME half-width rule inference
  uses (half = size*context/8), so train and test see one distribution.

Output layout matches scripts/train_junction.py exactly
({train,val}/{junction,crossover}/*.png + index.csv + dataset_meta.json),
so training is a drop-in:

    python scripts/train_junction.py --data data/crossings_synth \
        --size 128 --out experiments/junction/synth128

Usage:
    python scripts/build_crossing_dataset.py --rounds 3 \
        --out data/crossings_synth
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from schematic2netlist.classes import class_terminals
from schematic2netlist.oracle_render import (
    _blocked_for, _cell, _route, terminal_sites)
from schematic2netlist.preprocess import project_bbox

COCO = ("data/digitize_hcd/extracted/Digitize-HCD Dataset/"
        "Component Symbol and Text Label Data/component_annotations.json")


# ---------------------------------------------------------------- layouts

def load_layouts(coco_path: str, transforms_path: str,
                 split_file: str) -> list[tuple[str, list[dict]]]:
    """(stem, components) per split image, boxes projected to the frame.

    Wire Crossover boxes are annotation marks, not components — excluded.
    """
    coco = json.load(open(coco_path))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    by_name = {i["file_name"]: i["id"] for i in coco["images"]}
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    transforms = json.load(open(transforms_path))

    out = []
    for name in (l.strip() for l in open(split_file) if l.strip()):
        stem = Path(name).stem
        meta = transforms.get(stem)
        iid = by_name.get(name)
        if meta is None or iid is None:
            continue
        comps = []
        for a in anns[iid]:
            cls = cats[a["category_id"]]
            if cls == "Wire Crossover":
                continue
            x, y, w, h = a["bbox"]
            cx, cy, bw, bh = project_bbox(meta, x, y, w, h)
            if bw <= 0 or bh <= 0:
                continue
            comps.append({"class": cls, "bbox": [cx, cy, bw, bh]})
        if len(comps) >= 3:
            out.append((stem, comps))
    return out


# ------------------------------------------------------------- topologies

def synth_topology(comps: list[dict], rng: random.Random) -> dict[int, list]:
    """Random net partition over all terminals -> {net_id: [entry, ...]}.

    Entries mirror oracle_render's routing input: pin, stub, comp index.
    GND terminals are pooled onto one net with high probability (real
    drawings share a ground rail); everything else joins nets of 2-5
    terminals with small-net bias, mirroring real net-size statistics.
    """
    terms = []
    for ci, c in enumerate(comps):
        sites = terminal_sites(c)
        n_t = min(max(1, class_terminals(c["class"])), len(sites))
        for i in range(n_t):
            px, py, (nx, ny) = sites[i]
            terms.append({"pin": (px, py),
                          "stub": (px + nx * 9, py + ny * 9),
                          "comp": ci,
                          "is_gnd": c["class"] == "GND"})
    rng.shuffle(terms)

    nets: dict[int, list] = {}
    nid = 0
    gnd = [t for t in terms if t["is_gnd"]]
    rest = [t for t in terms if not t["is_gnd"]]
    if gnd and rng.random() < 0.8:
        # ground rail: all GND pins + a few extra terminals
        extra = [rest.pop() for _ in range(min(len(rest) - 2,
                                               rng.randint(1, 3)))
                 if len(rest) > 2]
        nets[nid] = gnd + extra
        nid += 1
    else:
        rest = terms

    # Group SPATIALLY NEAR terminals, not random ones. Random assignment
    # connects terminals on opposite sides of the page, so Lee routing
    # crisscrosses the whole canvas and the rendered patches come out far
    # denser than real ones: measured ink density 0.150 against 0.060 on
    # real inference patches (Cohen's d 1.79), which is a domain gap large
    # enough on its own to explain a classifier scoring 0.91 in-domain and
    # chance on real masks. Real schematics wire neighbours together.
    while len(rest) >= 2:
        k = rng.choices((2, 3, 4, 5), weights=(0.55, 0.25, 0.15, 0.05))[0]
        k = min(k, len(rest))
        if len(rest) - k == 1:          # never strand a single terminal
            k += 1
        seed_t = rest.pop(rng.randrange(len(rest)))
        sx, sy = seed_t["pin"]
        rest.sort(key=lambda t: (t["pin"][0] - sx) ** 2 + (t["pin"][1] - sy) ** 2)
        group = [seed_t]
        # nearest k-1, with a little jitter so layouts are not deterministic
        pool = rest[: min(len(rest), (k - 1) * 3)]
        rng.shuffle(pool)
        for t in pool[: k - 1]:
            group.append(t)
            rest.remove(t)
        nets[nid] = group
        nid += 1
    if rest and nets:                    # leftover single joins some net
        nets[rng.choice(list(nets))].append(rest.pop())
    return {k: v for k, v in nets.items() if len(v) >= 2}


def add_touch_contacts(routed: dict[int, set], rng, rate: float):
    """Make some nets TOUCH without connecting — the dominant real case.

    Causal analysis of welds (scripts/locate_welds.py) found 86% have a
    single intersection cut point and **41% of those cut points are
    degree-3**, i.e. a place where one net's stroke lands on another's and
    thick ink fuses them. Nothing in a maze-routed render produces that
    geometry, so a classifier trained without it never learns the case
    that matters most — every mechanism we built assumed degree-3 means
    "junction, by definition".

    Here a randomly chosen net is extended by a short stub until it meets
    another net's path. The two remain DIFFERENT nets, so the site is
    labelled as needing a split, with T geometry rather than X.
    """
    extra: list[tuple[int, int]] = []      # (x, y) contact points
    nets = sorted(routed)
    for a in nets:
        if rng.random() > rate:
            continue
        b = rng.choice([n for n in nets if n != a]) if len(nets) > 1 else None
        if b is None:
            continue
        cells_a = sorted(routed[a])
        cells_b = sorted(routed[b])
        if not cells_a or not cells_b:
            continue
        # nearest pair of cells between the two nets
        ay, ax, as_ = rng.choice(cells_a)
        best = min(cells_b, key=lambda c: (c[0] - ay) ** 2 + (c[1] - ax) ** 2)
        by, bx, bs = best
        if (by - ay) ** 2 + (bx - ax) ** 2 > 120 ** 2:
            continue                        # too far to be a plausible touch
        # walk a short L-shaped stub from a toward b, painted as net a
        y, x = ay, ax
        step = as_
        guard = 0
        while (y, x) != (by, bx) and guard < 60:
            if abs(bx - x) >= abs(by - y) and bx != x:
                x += step if bx > x else -step
            elif by != y:
                y += step if by > y else -step
            else:
                break
            routed[a].add((y, x, step))
            guard += 1
        extra.append((bx + bs // 2, by + bs // 2))
    return extra


def route_nets(comps, nets, shape, rng) -> dict[int, set]:
    """Route each net independently (nets may cross) -> {net: cell set}."""
    stride = 3
    h, w = shape
    routed: dict[int, set] = {}
    for net_id, entries in nets.items():
        owners = {e["comp"] for e in entries}
        for pad_try, stride_try in ((5, stride), (2, 2), (0, 2)):
            gshape = (h // stride_try + 1, w // stride_try + 1)
            blocked = _blocked_for(owners, comps, shape, stride_try, pad_try)
            cells = [_cell(e["stub"], stride_try, gshape) for e in entries]
            tree = {cells[0]}
            ok = True
            for tgt in cells[1:]:
                if tgt in tree:
                    continue
                path = _route(tgt, tree, blocked)
                if path is None:
                    ok = False
                    break
                tree.update(path)
            if ok:
                routed[net_id] = {(r * stride_try, c * stride_try, stride_try)
                                  for r, c in tree}
                break
    return routed


# ---------------------------------------------------------------- sites

def find_sites(routed: dict[int, set], min_sep: int = 14):
    """Exact crossing and junction sites from the routed cell geometry."""
    # crossings: same canvas position claimed by >=2 nets
    pos_nets: dict[tuple[int, int], set] = defaultdict(set)
    for net, cells in routed.items():
        for (y, x, s) in cells:
            pos_nets[(y // 6, x // 6)].add(net)   # coarse bin merges strides
    crossings = [(x * 6 + 3, y * 6 + 3)
                 for (y, x), ns in pos_nets.items() if len(ns) >= 2]

    # junctions: degree>=3 within one net's own tree
    junctions = []
    for net, cells in routed.items():
        grid = {(y // s, x // s): s for (y, x, s) in cells}
        for (r, c), s in grid.items():
            deg = sum(((r + dr, c + dc) in grid)
                      for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            if deg >= 3:
                junctions.append((c * s + s // 2, r * s + s // 2))

    def dedupe(pts):
        kept = []
        for p in pts:
            if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= min_sep ** 2
                   for q in kept):
                kept.append(p)
        return kept

    cross = dedupe(crossings)
    junc = [p for p in dedupe(junctions)
            if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= min_sep ** 2
                   for q in cross)]
    return cross, junc


# -------------------------------------------------------------- rendering

def render_ink(shape, comps, routed, cross_sites, junc_sites, rng,
               p_dot=0.45, p_hop=0.12):
    """Hand-drawn-style binary ink from the routed geometry."""
    h, w = shape
    ink = np.zeros((h, w), np.uint8)

    # low-frequency wobble field (one per image)
    fy, fx = rng.uniform(1.5, 4.0), rng.uniform(1.5, 4.0)
    ph1, ph2 = rng.uniform(0, 6.28), rng.uniform(0, 6.28)
    amp = rng.uniform(0.8, 2.2)

    def wob(x, y):
        return (x + amp * math.sin(2 * math.pi * fy * y / h + ph1),
                y + amp * math.sin(2 * math.pi * fx * x / w + ph2))

    hop_at = {s: None for s in cross_sites if rng.random() < p_hop}

    for net, cells in routed.items():
        # thinner than the original (2,3,3,4,5): real 1024-px masks run
        # ~3.8 px wide, and the heavier mix pushed patch ink density to
        # 0.150 against 0.060 measured on real inference patches
        t = rng.choice((2, 2, 3, 3, 4))
        grid = {(y // s, x // s): s for (y, x, s) in cells}
        for (r, c), s in grid.items():
            for dr, dc in ((1, 0), (0, 1)):
                if (r + dr, c + dc) not in grid:
                    continue
                x0, y0 = c * s + s // 2, r * s + s // 2
                x1, y1 = (c + dc) * s + s // 2, (r + dr) * s + s // 2
                if rng.random() < 0.004:          # pen lift
                    continue
                skip = False
                for (hx, hy) in hop_at:
                    if (x0 - hx) ** 2 + (y0 - hy) ** 2 < 10 ** 2:
                        if hop_at[(hx, hy)] is None:
                            hop_at[(hx, hy)] = net    # first net hops
                        if hop_at[(hx, hy)] == net:
                            skip = True               # gap under the hop
                            break
                if skip:
                    continue
                a = tuple(int(round(v)) for v in wob(x0, y0))
                b = tuple(int(round(v)) for v in wob(x1, y1))
                cv2.line(ink, a, b, 255, t)

    # hop arcs for the gapped wire
    for (hx, hy), net in hop_at.items():
        if net is None:
            continue
        r = rng.randint(5, 8)
        cv2.ellipse(ink, (hx, hy), (r, r), 0, 180, 360, 255, 2)

    # Junction dots, sized RELATIVE to the local stroke.
    #
    # These were drawn at a fixed radius 3-5 px while strokes are 2-5 px
    # wide, so a dot was frequently SMALLER than the wire it sat on and
    # left no trace. Measured on the first build: local blob radius was
    # 6.47 +- 1.56 px at junctions vs 5.90 +- 1.54 at crossings — an
    # effect size of ~0.37, i.e. no usable cue, which would have left the
    # classifier nothing to learn on exactly the ambiguous sites that
    # matter. A real solder dot is ~2x the conductor width, so measure the
    # local width and scale to it.
    if junc_sites:
        dt = cv2.distanceTransform((ink > 0).astype(np.uint8), cv2.DIST_L2, 3)
        for (jx, jy) in junc_sites:
            if rng.random() >= p_dot:
                continue
            y0, y1 = max(0, jy - 6), min(h, jy + 7)
            x0, x1 = max(0, jx - 6), min(w, jx + 7)
            local = dt[y0:y1, x0:x1]
            local_hw = float(local.max()) if local.size else 2.0
            # 1.15-1.6x, not 1.8-2.6x: the wider range produced a mean
            # blob radius of 9.53 px against 5.82 px measured on real
            # patches, overshooting the invisible-dot problem it fixed.
            r = int(round(max(3.0, local_hw * rng.uniform(1.15, 1.6))))
            cv2.circle(ink, (jx, jy), r, 255, -1)

    # erase component interiors exactly as the pipeline does (pad 0)
    for c in comps:
        cx, cy, bw, bh = c["bbox"]
        x1, y1 = max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2))
        x2, y2 = min(w, int(cx + bw / 2)), min(h, int(cy + bh / 2))
        ink[y1:y2, x1:x2] = 0
    return ink


def net_label_map(shape, routed, comps):
    """Per-net id at every routed pixel (-1 elsewhere); component
    interiors erased exactly as the pipeline erases them."""
    h, w = shape
    m = np.full((h, w), -1, np.int32)
    for net, cells in routed.items():
        for (y, x, s) in cells:
            cv2.rectangle(m, (x, y), (x + s - 1, y + s - 1), int(net), -1)
    for c in comps:
        cx, cy, bw, bh = c["bbox"]
        x1, y1 = max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2))
        x2, y2 = min(w, int(cx + bw / 2)), min(h, int(cy + bh / 2))
        m[y1:y2, x1:x2] = -1
    return m


def label_pipeline_sites(ink, nmap, radius: int = 14):
    """Label the sites the PIPELINE will actually ask about.

    Truth comes from the net map — two or more distinct nets in the
    surrounding ink means the site must be SPLIT, one net means UNION — so
    labels stay exact. What changes is the site POPULATION: the skeleton's
    own branch points rather than the router's cell overlaps. Verified
    (scripts/verify_crossing_labels.py) that the router-derived population
    covered only 77.5% of skeleton sites and skewed the wrong way on
    degree.
    """
    from schematic2netlist.skeleton import intersection_sites_with_degree

    H, W = ink.shape
    out = []
    for (x, y, deg) in intersection_sites_with_degree(
            (ink > 0).astype(np.uint8)):
        y0, y1 = max(0, y - radius), min(H, y + radius + 1)
        x0, x1 = max(0, x - radius), min(W, x + radius + 1)
        win = (ink[y0:y1, x0:x1] > 0).astype(np.uint8)
        # Read nets only from ink CONNECTED to the site, not everything in
        # the window: a net passing 10 px away without touching is not part
        # of this site, and counting it would label a plain junction as a
        # crossing. Connectivity is the electrical question, radius is not.
        n_lab, lab = cv2.connectedComponents(win, connectivity=8)
        sy, sx = min(y - y0, win.shape[0] - 1), min(x - x0, win.shape[1] - 1)
        here = int(lab[sy, sx])
        if here == 0:                       # site pixel itself is background
            ys, xs = np.nonzero(win)
            if ys.size == 0:
                continue
            k = int(np.argmin((ys - sy) ** 2 + (xs - sx) ** 2))
            here = int(lab[ys[k], xs[k]])
        sub_ink = lab == here
        present = set(np.unique(nmap[y0:y1, x0:x1][sub_ink])) - {-1}
        if not present:
            continue
        out.append((x, y, deg,
                    "crossover" if len(present) >= 2 else "junction"))
    return out


def crop(ink, x, y, half, size):
    H, W = ink.shape
    x1, y1, x2, y2 = x - half, y - half, x + half, y + half
    pl, pt = max(0, -x1), max(0, -y1)
    pr, pb = max(0, x2 - W), max(0, y2 - H)
    p = ink[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
    if p.size == 0:
        return None
    if any((pl, pt, pr, pb)):
        p = cv2.copyMakeBorder(p, pt, pb, pl, pr,
                               cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(p, (size, size), interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coco", default=COCO)
    ap.add_argument("--transforms", default="data/transforms_1024.json")
    ap.add_argument("--frame", type=int, default=1024)
    ap.add_argument("--rounds", type=int, default=3,
                    help="independent topologies per layout")
    ap.add_argument("--size", type=int, default=128, help="patch px")
    ap.add_argument("--context", type=float, default=3.0,
                    help="must match nodes.junction_context at inference")
    ap.add_argument("--p-dot", type=float, default=0.45)
    ap.add_argument("--touch-rate", type=float, default=0.8,
                    help="per-net chance of being extended to TOUCH another "
                         "net without connecting (synthesises the degree-3 "
                         "contact that is 41%% of real weld cut points)")
    ap.add_argument("--merge-q", type=float, default=0.4,
                    help="fraction of crossing net-pairs merged into "
                         "one net (their overlaps become junctions)")
    ap.add_argument("--p-hop", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip this many layouts first. With --limit this "
                         "shards the work, so N processes can generate "
                         "disjoint slices into the same output tree "
                         "(filenames are unique per layout+round). Lee-maze "
                         "routing is single-threaded and CPU-bound, so "
                         "sharding is the whole speedup on a multicore box.")
    ap.add_argument("--no-index", action="store_true",
                    help="skip index.csv/dataset_meta.json (shard workers)")
    ap.add_argument("--out", default="data/crossings_synth")
    args = ap.parse_args()

    half = max(4, int(round(args.size * args.context / 8)))
    out = Path(args.out)
    for split in ("train", "val"):
        for cls in ("junction", "crossover"):
            (out / split / cls).mkdir(parents=True, exist_ok=True)

    counts: dict[tuple[str, str], int] = defaultdict(int)
    index = []
    for split in ("train", "val"):
        layouts = load_layouts(args.coco, args.transforms,
                               f"data/splits/{split}.txt")
        layouts = layouts[args.offset:]
        if args.limit:
            layouts = layouts[: args.limit]
        print(f"{split}: {len(layouts)} layouts x {args.rounds} rounds")
        for li, (stem, comps) in enumerate(layouts):
            for rd in range(args.rounds):
                rng = random.Random((args.seed, split, stem, rd).__hash__())
                shape = (args.frame, args.frame)
                nets = synth_topology(comps, rng)
                routed = route_nets(comps, nets, shape, rng)
                if len(routed) < 2:
                    continue
                # T-contacts BEFORE merging, so a contact between two nets
                # that later merge correctly becomes a junction instead
                add_touch_contacts(routed, rng, args.touch_rate)
                # Merge a fraction of crossing net-pairs into ONE net:
                # their overlap points become junctions (a net crossing
                # itself IS a junction), which manufactures the exact
                # confusable case inference faces — same X geometry,
                # different electrical truth — at a controllable rate.
                # Without this, maze trees almost never branch and the
                # junction class starves (measured 13.6:1 the wrong way).
                pos_nets: dict[tuple[int, int], set] = defaultdict(set)
                for net, cells in routed.items():
                    for (y, x, s) in cells:
                        pos_nets[(y // 6, x // 6)].add(net)
                pairs = sorted({tuple(sorted(ns)[:2])
                                for ns in pos_nets.values() if len(ns) >= 2})
                group = {n: n for n in routed}

                def find(n):
                    while group[n] != n:
                        group[n] = group[group[n]]
                        n = group[n]
                    return n

                for a, b in pairs:
                    if rng.random() < args.merge_q and find(a) != find(b):
                        group[find(a)] = find(b)
                merged: dict[int, set] = defaultdict(set)
                for net, cells in routed.items():
                    merged[find(net)] |= cells
                cross, junc = find_sites(dict(merged))
                ink = render_ink(shape, comps, merged, cross, junc, rng,
                                 p_dot=args.p_dot, p_hop=args.p_hop)
                # Label the sites the SKELETON reports, not the ones the
                # router produced: emitting from routed geometry covered
                # only 77.5% of the sites intersection_sites_with_degree
                # finds on the same render, and the missing 22.5% skewed
                # degree-3 (53 of 93) while training skewed degree-4
                # (242 vs 76) — under-covering exactly the T-sites that are
                # 41% of real weld cut points. Truth still comes from the
                # net map, so labels stay exact; only the site POPULATION
                # changes, and it now equals the inference population.
                nmap = net_label_map(shape, merged, comps)
                for si, (x, y, deg, cls) in enumerate(
                        label_pipeline_sites(ink, nmap)):
                    p_img = crop(ink, x, y, half, args.size)
                    if p_img is None or (p_img > 0).mean() < 0.02:
                        continue
                    name = f"{stem}__r{rd}__{si}.png"
                    cv2.imwrite(str(out / split / cls / name), p_img)
                    counts[(split, cls)] += 1
                    index.append({"file": f"{split}/{cls}/{name}",
                                  "split": split, "class": cls,
                                  "drafter": stem, "source": stem,
                                  "box": [x, y], "degree": deg})
            if (li + 1) % 100 == 0:
                print(f"  [{li + 1}/{len(layouts)}] "
                      f"{sum(counts.values())} patches", flush=True)

    if args.no_index:
        print(f"shard done: {sum(counts.values())} patches "
              f"(offset={args.offset} limit={args.limit})")
        return
    with (out / "index.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index[0].keys()))
        w.writeheader()
        w.writerows(index)
    meta = {"source": "synthetic routed renders over real train/val layouts",
            "frame": args.frame, "patch_size": args.size,
            "context": args.context, "half": half, "rounds": args.rounds,
            "p_dot": args.p_dot, "p_hop": args.p_hop, "seed": args.seed,
            "counts": {f"{s}/{c}": n for (s, c), n in sorted(counts.items())},
            "test_split_touched": False}
    (out / "dataset_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta["counts"], indent=2))
    print(f"wrote {sum(counts.values())} patches to {out}")


if __name__ == "__main__":
    main()
