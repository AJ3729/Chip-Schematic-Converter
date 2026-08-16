* Auto-generated SPICE netlist (NO TEXT OCR USED)

L1 0 n2 1m
C1 n4 n2 1u
C2 0 n2 1u
L2 n4 0 1m
* SAME_NODE_SKIPPED Resistor both_on=n1
C3 n3 n4 1u

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n3 n3 0 1e+09

.op
.end
