* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 n2 0 AC 1m
* SAME_NODE_SKIPPED I-DC both_on=0
R1 0 n1 1k
L1 n1 0 1m

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09

.op
.end
