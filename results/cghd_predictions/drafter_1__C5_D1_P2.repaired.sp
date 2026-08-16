* Auto-generated SPICE netlist (NO TEXT OCR USED)

V1 n4 n2 DC 5
R1 n3 n4 1k
R2 n2 n3 1k
* SAME_NODE_SKIPPED I-DC both_on=n1
* SAME_NODE_SKIPPED I-DC both_on=0
* SAME_NODE_SKIPPED V-AC both_on=0

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09

.op
.end
