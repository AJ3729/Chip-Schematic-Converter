"""Turn a decisions file into the final GT JSON (verified=false, annotator=null)."""
from __future__ import annotations
import json, os, sys, copy
sys.path.insert(0, "/home/claude/tools")
from netbuild import analyse
from erc import check

OUT_GT = "/home/claude/out/gt"


def finalize(stem, root, decisions, outdir=OUT_GT):
    gt = json.load(open(f"{root}/gt/{stem}.json"))
    img = f"{root}/img1024/{stem}.jpg"
    # class corrections first — they change terminal counts
    classes = {int(k): v for k, v in (decisions.get("classes") or {}).items()}
    from erc import TERMS
    for c in gt["components"]:
        if c["id"] in classes:
            c["class"] = classes[c["id"]]
            want = TERMS[c["class"]]
            if len(c["terminals"]) != want:
                c["terminals"] = [{"index": i, "net": None} for i in range(want)]
    for extra in (decisions.get("add_components") or []):
        gt["components"].append({"id": extra["id"], "class": extra["class"],
                                 "bbox": extra["bbox"],
                                 "terminals": [{"index": i, "net": None}
                                               for i in range(TERMS[extra["class"]])]})
    for cid in (decisions.get("remove_components") or []):
        gt["components"] = [c for c in gt["components"] if c["id"] != cid]

    res = analyse(img, gt, decisions)
    out = copy.deepcopy(gt)
    out["source"] = "coco_geometry+manual_topology"
    out["verified"] = False
    out["annotator"] = None
    out["bbox_frame"] = "cleaned_1024"
    out["notes"] = decisions.get("notes", "")
    unconn = set(decisions.get("unconnected") or [])
    for c in out["components"]:
        for t in c["terminals"]:
            t["net"] = res["nets"].get((c["id"], t["index"]))
        if c["id"] in unconn:
            c["unconnected"] = True
        else:
            c.pop("unconnected", None)
    os.makedirs(outdir, exist_ok=True)
    p = f"{outdir}/{stem}.json"
    json.dump(out, open(p, "w"), indent=2)
    errs, warns = check(out)
    return out, errs, warns, res


if __name__ == "__main__":
    stem = sys.argv[1]
    decp = sys.argv[2]
    root = sys.argv[3] if len(sys.argv) > 3 else "/home/claude/val"
    dec = json.load(open(decp)) if os.path.exists(decp) else {}
    out, errs, warns, res = finalize(stem, root, dec)
    print(f"wrote {OUT_GT}/{stem}.json  ({len(out['components'])} components, "
          f"{len(set(v for c in out['components'] for v in [t['net'] for t in c['terminals']] if v))} nets)")
    for c in out["components"]:
        print(f"  #{c['id']:<3} {c['class']:<16} " +
              " ".join(f"t{t['index']}={t['net']}" for t in c["terminals"]) +
              ("  UNCONNECTED" if c.get("unconnected") else ""))
    if errs:
        print("ERRORS:")
        for e in errs:
            print("   [ERR ]", e)
    if warns:
        print("WARNINGS:")
        for w in warns:
            print("   [warn]", w)
    if not errs and not warns:
        print("ERC clean.")
