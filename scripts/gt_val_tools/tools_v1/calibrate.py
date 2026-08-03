"""Calibrate the tracer against verified GT.

Grouping quality is measured up to a per-component permutation of terminal
indices (terminal ORDER is a separate, vision-level question), so the number
reported here is purely about which terminals share a net.
"""
import json, glob, os, sys, itertools, random
sys.path.insert(0, "/home/claude/tools")
from trace import trace, assign_terminals, nets_from


def best_perm_accuracy(gtn, auto, comps):
    """gtn/auto: {(cid,ti): net}. Optimise per-component index permutation."""
    keys = [k for k in gtn if gtn[k] is not None and k in auto]
    if len(keys) < 2:
        return 1.0, 0, 0, 0
    by_c = {}
    for (cid, ti) in keys:
        by_c.setdefault(cid, []).append(ti)
    perm = {cid: {t: t for t in tis} for cid, tis in by_c.items()}

    def agree(perm):
        ok = 0; tot = 0; osp = 0; omg = 0
        mapped = {(c, perm[c][t]): auto[(c, t)] for (c, t) in keys}
        gm = {(c, t): gtn[(c, t)] for (c, t) in keys}
        ks = list(gm)
        for a, b in itertools.combinations(ks, 2):
            tot += 1
            same_gt = gm[a] == gm[b]
            same_au = mapped[a] == mapped[b]
            if same_gt == same_au:
                ok += 1
            elif same_gt:
                osp += 1
            else:
                omg += 1
        return ok, tot, osp, omg

    ok, tot, osp, omg = agree(perm)
    improved = True
    rounds = 0
    while improved and rounds < 6:
        improved = False; rounds += 1
        for cid, tis in by_c.items():
            if len(tis) < 2:
                continue
            base = perm[cid]
            best = (ok, base)
            for p in itertools.permutations(tis):
                cand = dict(zip(tis, p))
                perm[cid] = cand
                o, t, _, _ = agree(perm)
                if o > best[0]:
                    best = (o, cand)
            perm[cid] = best[1]
            if best[0] > ok:
                ok = best[0]; improved = True
    ok, tot, osp, omg = agree(perm)
    nperm = sum(1 for cid, m in perm.items() if any(k != v for k, v in m.items()))
    return ok / tot, osp, omg, nperm


def run(root, limit=10**9, verbose=True, **kw):
    files = sorted(glob.glob(f"{root}/gt/*.json"))[:limit]
    tot_t = cov_t = 0
    accs = []; osp_t = omg_t = 0; exact = 0; per = []
    for f in files:
        gt = json.load(open(f)); stem = os.path.basename(f)[:-5]
        try:
            tr = trace(f"{root}/img1024/{stem}.jpg", gt, **kw)
            asg = assign_terminals(gt, tr)
            auto, _ = nets_from(gt, tr, asg)
        except Exception as e:
            per.append((stem, 0.0, 0.0, "ERR " + str(e)[:70])); continue
        gtn = {(c["id"], t["index"]): t["net"] for c in gt["components"] for t in c["terminals"]}
        keys = [k for k, v in gtn.items() if v is not None]
        tot_t += len(keys); cov = [k for k in keys if k in auto]; cov_t += len(cov)
        acc, osp, omg, nperm = best_perm_accuracy(gtn, auto, gt["components"])
        accs.append(acc); osp_t += osp; omg_t += omg
        good = (len(cov) == len(keys)) and acc == 1.0
        exact += good
        per.append((stem, len(cov) / max(1, len(keys)), acc, "OK" if good else ""))
    n = max(1, len(files))
    if verbose:
        print(f"images {len(files)}  exact(group+cov) {exact} ({exact/n:.1%})")
        print(f"terminal coverage {cov_t}/{tot_t} = {cov_t/max(1,tot_t):.1%}")
        print(f"mean grouping pair-acc {sum(accs)/max(1,len(accs)):.4f}")
        print(f"over-split pairs {osp_t}  over-merge pairs {omg_t}")
        bad = sorted([p for p in per if p[3] != "OK"], key=lambda p: p[2])[:15]
        print("worst:", *[f"{b[0]} cov={b[1]:.2f} grp={b[2]:.3f} {b[3]}" for b in bad], sep="\n  ")
    return {"exact": exact, "n": len(files), "cov": cov_t / max(1, tot_t),
            "acc": sum(accs) / max(1, len(accs)), "osp": osp_t, "omg": omg_t, "per": per}


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/cal"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
    run(root, limit)
