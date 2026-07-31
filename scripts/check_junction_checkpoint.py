#!/usr/bin/env python3
"""Do a returned checkpoint and its summary.json actually describe the same
model?

A training run writes three artefacts at three different moments:
``best.pt`` and ``val_probs.npy`` whenever validation improves, and
``summary.json`` once at the end. A second run into the same output
directory overwrites the first two at its own epoch 1 — ``best`` resets to
-1.0, so the first epoch always "improves" — while ``summary.json`` still
describes the finished run until this one completes. Interrupt it and the
directory ships a finished run's metrics next to a half-trained run's
weights.

That is not hypothetical. The v5 pod run returned exactly this pair:
summary.json reporting balanced accuracy 0.8019 at epoch 57, alongside
weights worth 0.6746. The mismatch is invisible from the filenames, and a
transfer evaluation ran on the wrong model before anyone noticed — the
result read as "the render domain does not transfer" when the 0.80
weights had never been scored at all.

The tell is cheap and label-free: the number of validation patches whose
probability clears a threshold depends only on the probabilities, not on
the labels. If ``summary.json``'s sweep says 6,900 samples clear 0.10 but
``val_probs.npy``'s minimum is 0.17 (so all 8,476 clear it), the two files
cannot describe one model.

This script runs three independent checks:

1. **best.pt vs val_probs.npy** — re-infer the validation split through
   the *pipeline's* inference path (``junction_model.load_model``, not the
   trainer's own loop) and compare to the saved probabilities. This also
   catches a train/inference skew: a normalization or architecture drift
   between the two paths would show up here as a large elementwise
   difference even when the files do match.
2. **val_probs.npy vs summary.json** — recompute the threshold sweep and
   compare predicted-positive counts, which are label-independent.
3. **in-domain AUC** — reported threshold-free, because balanced accuracy
   at a fixed 0.5 hides how much ranking signal a model actually has.

Exit status is non-zero when the artefacts disagree, so this can gate an
evaluation rather than merely inform one.

Usage:
    python scripts/check_junction_checkpoint.py \
        --run experiments/junction/synth128_v5 --data /tmp/crossings_v5.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def auc_score(pos: np.ndarray, neg: np.ndarray) -> float:
    import scipy.stats as ss
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = ss.rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (
        len(pos) * len(neg))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True,
                    help="directory holding best.pt / summary.json")
    ap.add_argument("--data", required=True,
                    help="the packed .npz the run was trained on")
    ap.add_argument("--seed", type=int, default=0,
                    help="must match the training --seed; the validation "
                         "split is permuted with it")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    import torch
    from schematic2netlist.junction_model import load_model

    run = Path(args.run)
    ok = True

    z = np.load(args.data, allow_pickle=False)
    Xva, yva = z["X_val"], z["y_val"]
    # the trainer permutes validation with default_rng(seed); val_probs.npy
    # is in that order, so reproduce it before comparing anything
    vperm = np.random.default_rng(args.seed).permutation(len(yva))
    Xva, yva = Xva[vperm], yva[vperm]
    print(f"data {args.data}: val {len(yva)} patches "
          f"({int((yva == 1).sum())} crossover)")

    model, size = load_model(str(run / "best.pt"))
    ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    stamp = ckpt.get("val_metrics")
    print(f"checkpoint: {size}px patches"
          + (f", epoch {ckpt['epoch']}, balanced_acc "
             f"{stamp['balanced_acc']:.4f} (stamped)" if stamp else
             ", NO metrics stamp (written before this was recorded)"))

    if Xva.shape[1] != size:
        print(f"\nFAIL: checkpoint wants {size}px patches, {args.data} holds "
              f"{Xva.shape[1]}px — wrong dataset for this run")
        return 1

    probs = []
    with torch.no_grad():
        for i in range(0, len(Xva), args.batch):
            xb = torch.from_numpy(Xva[i:i + args.batch]).unsqueeze(1)
            probs.append(
                torch.softmax(model(xb.float().div_(255.0)), 1)[:, 1].numpy())
    probs = np.concatenate(probs)

    # --- 1. best.pt vs val_probs.npy -------------------------------------
    vp_path = run / "val_probs.npy"
    if vp_path.exists():
        vp = np.load(vp_path)
        if vp.shape != probs.shape:
            print(f"\nFAIL check 1: val_probs.npy has {vp.shape}, validation "
                  f"split has {probs.shape}")
            ok = False
        else:
            dev = float(np.abs(vp - probs).max())
            # float32 across CUDA/CPU drifts ~1e-4; anything larger means
            # different weights or a different inference path
            verdict = "OK" if dev < 5e-3 else "FAIL"
            ok &= dev < 5e-3
            print(f"\n[{verdict}] check 1  best.pt reproduces val_probs.npy: "
                  f"max elementwise diff {dev:.2e}")
            if dev >= 5e-3:
                print("        best.pt and val_probs.npy are from different "
                      "models, or the pipeline inference path has drifted "
                      "from the trainer's.")
    else:
        print("\n[skip] check 1  no val_probs.npy in the run directory")

    # --- 2. val_probs.npy vs summary.json --------------------------------
    sm_path = run / "summary.json"
    if sm_path.exists():
        sm = json.loads(sm_path.read_text())
        # Exact equality is the WRONG test. The trainer counts on the GPU in
        # float32 and this re-inference runs on the CPU, so a handful of
        # probabilities sitting within float noise of a threshold legitimately
        # flip sides. A tolerance separates the two failure scales cleanly:
        # the genuine mismatch this script was written for was off by 290-1576
        # of 8476 (up to 18.6%) with balanced accuracy off by 0.13, while
        # CUDA-vs-CPU drift on the same weights is off by single digits
        # (<0.1%) with balanced accuracy matching to 3e-04.
        tol = max(4, int(0.01 * len(probs)))
        bad, worst = [], 0
        for r in sm.get("threshold_sweep", []):
            t = r["threshold"]
            reported = r["tp"] + r["fp"]          # label-independent
            actual = int((probs >= t).sum())
            worst = max(worst, abs(reported - actual))
            if abs(reported - actual) > tol:
                bad.append((t, reported, actual))
        if not bad and worst:
            print(f"\n[OK]   check 2  summary.json matches best.pt within "
                  f"float tolerance (worst predicted-positive discrepancy "
                  f"{worst} of {len(probs)} = {worst/len(probs):.2%}, "
                  f"tolerance {tol})")
        if bad:
            ok = False
            print(f"\n[FAIL] check 2  summary.json describes different "
                  f"probabilities than best.pt produces "
                  f"({len(bad)}/{len(sm['threshold_sweep'])} thresholds "
                  f"disagree on the predicted-positive count, which does not "
                  f"depend on labels at all):")
            print(f"        {'thr':>5s} {'summary':>9s} {'best.pt':>9s}")
            for t, reported, actual in bad[:6]:
                print(f"        {t:5.2f} {reported:9d} {actual:9d}")
            b = sm.get("best_at_0.5", {})
            if b:
                print(f"        summary reports balanced_acc "
                      f"{b.get('balanced_acc')} at epoch {b.get('epoch')}; "
                      f"these weights are worth something else.")
            print("        Re-tar the run directory, or retrain into a fresh "
                  "--out. Do NOT evaluate these weights as if they were the "
                  "reported model.")
        elif not worst:
            print("\n[OK]   check 2  summary.json matches best.pt exactly at "
                  "every swept threshold")
        for k in ("data", "val_counts"):
            if k in sm:
                print(f"        summary {k}: {sm[k]}")
    else:
        print("\n[skip] check 2  no summary.json in the run directory")

    # --- 3. in-domain ranking signal -------------------------------------
    a = auc_score(probs[yva == 1], probs[yva == 0])
    pred = probs >= 0.5
    tp = int((pred & (yva == 1)).sum()); fn = int((~pred & (yva == 1)).sum())
    fp = int((pred & (yva == 0)).sum()); tn = int((~pred & (yva == 0)).sum())
    bal = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
    print(f"\n[info] check 3  these weights, measured directly:")
    print(f"        in-domain AUC          {a:.4f}")
    print(f"        balanced_acc @ 0.5     {bal:.4f}  "
          f"(tp {tp} fp {fp} fn {fn} tn {tn})")
    if stamp and abs(stamp["balanced_acc"] - bal) > 5e-3:
        ok = False
        print(f"        [FAIL] the checkpoint's own stamp says "
              f"{stamp['balanced_acc']:.4f} — it was saved at a different "
              f"epoch than these weights")

    print("\n" + ("ARTEFACTS CONSISTENT — safe to evaluate"
                  if ok else
                  "ARTEFACTS INCONSISTENT — fix before evaluating"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
