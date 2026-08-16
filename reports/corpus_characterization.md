# Corpus characterization (task B4)

Measured on the **original photographs**, not the rectified frames —
rectification exists precisely to remove these differences, so
measuring the frames would hide what the corpora actually differ in.

Digitize-HCD: 250 images. CGHD: 250 images (v12).

## Side by side

| property | Digitize-HCD | CGHD |
| --- | --- | --- |
| resolution (MP) (median [p10–p90]) | 2.403 [1.502–5.564] | 2.286 [1.097–12.193] |
| aspect ratio w/h (median [p10–p90]) | 1.785 [1.021–2.234] | 1.333 [0.75–1.778] |
| shadow field strength (median [p10–p90]) | 3.44 [2.617–4.428] | 21.622 [8.054–45.296] |
| background variation (median [p10–p90]) | 2.316 [0.88–42.79] | 31.763 [13.093–66.059] |
| JPEG quality (est.) (median [p10–p90]) | 88.6 [88.6–88.6] | 97.2 [86.0–99.5] |
| portrait fraction | 0.092 | 0.16 |
| images with EXIF camera | 0 | 169 |

## Conclusion

Derived from the table above, ranked by how far each property differs
(|log ratio|), rather than asserted:

- **background variation**: CGHD is 13.71x — 1271% higher than Digitize-HCD.
- **shadow field strength**: CGHD is 6.29x — 529% higher than Digitize-HCD.
- **aspect ratio w/h**: CGHD is 0.75x — 25% lower than Digitize-HCD.
- **JPEG quality (est.)**: CGHD is 1.10x — 10% higher than Digitize-HCD.
- **resolution (MP)**: CGHD is 0.95x — 5% lower than Digitize-HCD.

**Sampling.** Both corpora are sampled uniformly at random with a
fixed seed. `drafter_0` is excluded from CGHD, as it is everywhere
else in this work: 917 of its 1,038 images carry no annotation, so it
appears in no result and characterising it would describe images the
evaluation never sees.

**Reading this against the transfer result.** The cross-corpus
detection drop tracks component scale (recall 0.183 in the smallest
size quintile against ~0.55 elsewhere). Whichever imaging properties
differ most above are candidates for *why* small components are hard —
compression and background clutter both destroy the fine strokes that
distinguish a small symbol — but this table establishes association,
not cause. No experiment here isolates one property.
