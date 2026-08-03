"""Upper bound: if every intersection-site call were made correctly, how good
would the extracted netlist be? Greedy hill-climb on site decisions against
verified GT. This measures the quality of the substrate (wire graph + ports),
which is what a human reviewer cannot fix by looking at crops."""
import json, glob, os, sys, itertools
sys.path.insert(0, "/home/claude/tools")
from netbuild import analyse
from calibrate import best_perm_accuracy


def score(gt, nets):
    gtn = {(c["id"], t["index"]): t["net"] for c in gt["components"] for t in c["terminals"]}
    acc, osp, omg, _ = best_perm_accuracy(gtn, nets, gt["components"])
    return acc, osp, omg


def oracle(root, stem, gt, max_rounds=3):
    dec = {"sites": {}}
    r = analyse(f"{root}/img1024/{stem}.jpg", gt, dec)
    acc, _, _ = score(gt, r["nets"])
    crit = [s for s in r["sites"] if s["critical"]]
    for _ in range(max_rounds):
        improved = False
        for s in crit:
            cur = dec["sites"].get(str(s["site"]))
            alt = "junction" if (cur or s["default"]) == "crossing" else "crossing"
            d2 = {"sites": dict(dec["sites"])}
            d2["sites"][str(s["site"])] = alt
            r2 = analyse(f"{root}/img1024/{stem}.jpg", gt, d2)
            a2, _, _ = score(gt, r2["nets"])
            if a2 > acc + 1e-9:
                acc = a2; dec = d2; improved = True
                r = r2
                crit = [x for x in r2["sites"] if x["critical"]] or crit
        if not improved:
            break
    return acc, dec, r


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/cal"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    accs = []; ex = 0; cov_n = cov_d = 0; flips = 0; ncrit = 0
    for f in sorted(glob.glob(f"{root}/gt/*.json"))[:limit]:
        gt = json.load(open(f)); stem = os.path.basename(f)[:-5]
        try:
            acc, dec, r = oracle(root, stem, gt)
        except Exception as e:
            print("ERR", stem, repr(e)[:80]); continue
        gtn = {(c["id"], t["index"]): t["net"] for c in gt["components"] for t in c["terminals"]}
        keys = [k for k, v in gtn.items() if v is not None]
        cov_d += len(keys); cov_n += sum(1 for k in keys if k in r["nets"])
        accs.append(acc); ex += (acc == 1.0 and all(k in r["nets"] for k in keys))
        flips += len(dec["sites"]); ncrit += sum(1 for s in r["sites"] if s["critical"])
        print(f"{stem:>16} oracle_acc={acc:.4f} flips={len(dec['sites'])} "
              f"crit={sum(1 for s in r['sites'] if s['critical'])}")
    n = max(1, len(accs))
    print(f"\nORACLE mean acc {sum(accs)/n:.4f}  exact {ex}/{n}  coverage {cov_n/max(1,cov_d):.3f}"
          f"  mean flips {flips/n:.1f}  mean critical sites {ncrit/n:.1f}")
