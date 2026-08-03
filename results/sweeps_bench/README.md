# Which split each sweep was scored on

`run_meta.json` in every value directory records a `"split"` field, and on
2026-08-03 two of the three names in this tree changed meaning. Read this
before quoting anything here.

| runs | `run_meta.json` says | n | dated | what it actually is |
| --- | --- | --- | --- | --- |
| 122 | `split: test`, `gt_dir: data/gt_1024` | 190 | 2026-07-30 | **the validation split.** Those 190 images were renamed `val` in the split swap, and `gt_1024` is now `gt_val_1024`. Correctly a selection sweep. |
| 15 | `split: val_sample50`, `gt_dir: data/gt_val` | 50 | 2026-08-02 | **50 images that are now part of the test split.** Both paths are gone: the manifest is `splits/test_sample50_preswap.txt` and the GT is `gt_val50_preswap/`. |

## The 15 runs touched what is now test data

They are the action-item-1 sweeps — `wires.min_blob_area`, `min_blob_extent`,
`binarize_k`, `binarize_window` — run when those 50 images were the only
annotated validation set available.

**No shipped parameter derives from them.** Every one came back inert or already
at its shipped value, so nothing was selected and no value moved:

    wires.min_blob_area      2..24    strict 0.4600 at every value   INERT
    wires.min_blob_extent    2..16    strict 0.4600 at every value   INERT
    wires.binarize_k      0.10..0.30  flat to 0.25, 0.44 at 0.30     at optimum
    wires.binarize_window   15..51    flat, shipped value best       at optimum

The whole blob filter is dead code either way: `clean_blobs` keeps a blob on
area **or** extent, so neither threshold ever binds. Full account in commit
`8a4cb255`.

They are kept because they are the evidence for that negative result, and
because a record of which images a sweep read is worth more than a clean-looking
tree. `sweep_param.py` now defaults to `--split val --gt-dir data/gt_val_1024`,
so scoring test takes saying so.

## Not committed here

`results/weld_adjudication/*.png` — 56 review crops, ~1 MB each, regenerable
with `scripts/adjudicate_welds.py`. The adjudication result itself is
`manifest.csv` in that directory and is versioned.
