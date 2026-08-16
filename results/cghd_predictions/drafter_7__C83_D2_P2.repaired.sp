* Auto-generated SPICE netlist (NO TEXT OCR USED)

I1 n1 n2 AC 1m
R1 n1 n2 1k
* SAME_NODE_SKIPPED I-DC both_on=0
R2 n3 n4 1k
I2 0 n3 AC 1m
R3 0 n1 1k

* --- design-intent repair (does not change topology) ---
Rshunt_n3 n3 0 1e+09

.op
.end
