* Auto-generated SPICE netlist (NO TEXT OCR USED)

L1 0 n2 1m
L2 n1 0 1m
* SAME_NODE_SKIPPED Inductor both_on=0
C1 n2 n3 1u
C2 n3 0 1u
C3 n3 0 1u
C4 n3 0 1u

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n3 n3 0 1e+09

.op
.end
