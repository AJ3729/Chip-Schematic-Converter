* Auto-generated SPICE netlist (NO TEXT OCR USED)

* SAME_NODE_SKIPPED Inductor both_on=0
L1 n2 0 1m
L2 0 n4 1m
C1 n5 0 1u
C2 n4 n5 1u
* SAME_NODE_SKIPPED Inductor both_on=n1
C3 n3 n2 1u

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n3 n3 0 1e+09
Rshunt_n5 n5 0 1e+09

.op
.end
